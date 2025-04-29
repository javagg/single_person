# single people detection
## download model
```
yolo export model=yolov8n.pt format=onnx
```
## run and test
```
cargo run --release
curl -X POST -F "file=@test.jpg" http://localhost:3030/detect
```