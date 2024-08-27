from yolov5 import YOLOv5


def predict(img, model_name):
    model = YOLOv5(f'{model_name}.pt')
    
    return model.predict(img)