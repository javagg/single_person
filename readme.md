# single people detection
## download model
```
pip install ultralytics
yolo export model=yolo11n.pt format=onnx
```
## run and test
```
cargo run --release
curl --data-binary "@grace_hopper.jpg" http://localhost:3030/detect
```