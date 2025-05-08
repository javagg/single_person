# single people detection
## download model
```
pip install ultralytics
yolo export model=yolo11n.pt format=onnx
```
## run and test
```
yolo detect predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg' save=True
cargo run --release
curl --data-binary "@grace_hopper.jpg" http://localhost:3030/detect
```