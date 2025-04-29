use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageFormat, Rgba};
use serde::Serialize;
use std::{convert::Infallible, sync::OnceLock};
use thiserror::Error;
use tract_onnx::prelude::*;
use warp::{Filter, Rejection, Reply};

static MODEL: OnceLock<
    RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
> = OnceLock::new();

fn init_model(
) -> &'static RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>> {
    MODEL.get_or_init(|| {
        tract_onnx::onnx()
            .model_for_path("yolov8n.onnx")
            .expect("Failed to load model")
            .into_optimized()
            .expect("Optimization failed")
            .into_runnable()
            .expect("Failed to make runnable")
    })
}
fn preprocess(img: &DynamicImage) -> Result<Tensor, Box<dyn std::error::Error>> {
    // 调整大小并转换为 RGB
    let img = img.resize_exact(640, 640, FilterType::CatmullRom);
    let rgb_img = img.to_rgb8();

    let input = tract_ndarray::Array4::from_shape_fn((1, 3, 640, 640), |(_, c, y, x)| {
        let pixel = rgb_img.get_pixel(x as u32, y as u32);
        (pixel[c] as f32) / 255.0 // 归一化到 [0,1]
    });

    Ok(Tensor::from(input))
}

fn postprocess(
    outputs: &TVec<TValue>,
    orig_img: &DynamicImage,
    debug: bool,
) -> Result<f32, Box<dyn std::error::Error>> {
    // YOLOv8 的输出格式为 (1, 84, 8400)
    let output = outputs[0].to_array_view::<f32>()?;
    let (orig_w, orig_h) = orig_img.dimensions();

    let mut detections = Vec::new();
    for i in 0..output.shape()[2] {
        let confidence = output[[0, 4, i]]; // 置信度在索引4
        if confidence < 0.5 {
            continue;
        } // 置信度阈值

        let x = output[[0, 0, i]] * orig_w as f32 / 640.0;
        let y = output[[0, 1, i]] * orig_h as f32 / 640.0;
        let w = output[[0, 2, i]] * orig_w as f32 / 640.0;
        let h = output[[0, 3, i]] * orig_h as f32 / 640.0;

        detections.push((x, y, w, h, confidence));
    }

    if debug {
        let mut img = orig_img.to_rgba8();
        for (x, y, w, h, _) in &detections {
            let x1 = (x - w / 2.0) as i32;
            let y1 = (y - h / 2.0) as i32;
            let x2 = (x + w / 2.0) as i32;
            let y2 = (y + h / 2.0) as i32;

            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(x1, y1).of_size((x2 - x1) as u32, (y2 - y1) as u32),
                Rgba([255, 0, 0, 255]),
            );
        }
        img.save("debug_output.png")?;
    }

    let prob = match detections.len() {
        0 => 0.0,
        1 => 1.0,
        _ => 0.0,
    };
    Ok(prob)
}

async fn detect_handler(data: bytes::Bytes) -> Result<impl Reply, Rejection> {
    // 验证图片格式
    // let format = ImageFormat::from_mime_type(&data)
    //     .map_err(|_| warp::reject::custom(ApiError::InvalidImageFormat))?;

    // if !matches!(format, ImageFormat::Jpeg | ImageFormat::Png) {
    //     return Err(warp::reject::custom(ApiError::InvalidImageFormat));
    // }

    let img = image::load_from_memory(&data)
        .map_err(|_| warp::reject::custom(ApiError::ImageDecodeFailure))?;
    let input = preprocess(&img).map_err(|_| warp::reject::reject())?;
    let outputs = MODEL
        .get()
        .expect("bad model")
        .run(tvec!(input.to_owned().into()))
        .map_err(|_| warp::reject::reject())?;
    let debug_mode = cfg!(debug_assertions); // 调试模式自动开启
    let prob = postprocess(&outputs, &img, debug_mode).map_err(|_| warp::reject::reject())?;

    #[derive(Serialize)]
    struct Response {
        prob: f32,
    }
    Ok(warp::reply::json(&Response { prob }))
}

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("文件大小超过限制 (最大 {0}MB)")]
    FileTooLarge(usize),

    #[error("不支持的图片格式 (支持: jpg, png)")]
    InvalidImageFormat,

    #[error("图片解码失败")]
    ImageDecodeFailure,

    #[error("内部服务错误")]
    InternalError(#[from] anyhow::Error),
}

impl warp::reject::Reject for ApiError {}

#[derive(Serialize)]
struct ErrorResponse {
    code: u16,
    message: String,
}

async fn handle_rejection(err: Rejection) -> Result<impl Reply, Infallible> {
    let (code, message) = if let Some(api_err) = err.find::<ApiError>() {
        match api_err {
            ApiError::FileTooLarge(max) => (413, format!("文件大小超过 {}MB", max)),
            ApiError::InvalidImageFormat => (415, "不支持的图片格式".to_string()),
            ApiError::ImageDecodeFailure => (400, "图片解码失败".to_string()),
            ApiError::InternalError(_) => (500, "内部服务错误".to_string()),
        }
    } else if err.find::<warp::reject::PayloadTooLarge>().is_some() {
        (413, "请求体过大".to_string())
    } else {
        (500, "未知错误".to_string())
    };

    let json = warp::reply::json(&ErrorResponse { code, message });

    Ok(warp::reply::with_status(
        json,
        warp::http::StatusCode::from_u16(code).unwrap(),
    ))
}

#[tokio::main]
async fn main() {
    init_model();

    let detect_route = warp::post()
        .and(warp::path("detect"))
        .and(warp::body::bytes())
        .and_then(detect_handler)
        .recover(handle_rejection);

    warp::serve(detect_route).run(([0, 0, 0, 0], 3030)).await;
}
