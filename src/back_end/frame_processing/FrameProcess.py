import cv2
from src.back_end.model_training.arch import *
import torch.nn as nn



def frame_process(model : nn.Module):
    cap = cv2.VideoCapture(0)  # 0 for default camera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.5)
        annotated_frame = results[0].plot()  # Draw detections
        cv2.imshow('Garbage Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = None
    frame_process(model)