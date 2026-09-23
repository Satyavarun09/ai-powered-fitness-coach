import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render, redirect
from datetime import datetime
from django.contrib import messages
from django.http import HttpResponse
from django.utils import timezone
import os
import base64
import pandas as pd
import numpy as np
import io
import pickle
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from .models import UserProfile
import hashlib

# ── Constants ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50
GREEN = (0, 255, 0)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'   # Change this to your preferred admin password

# Lazy-load YOLO model (only when first needed)
_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("model/best.pt")
        print("Yolo Model Loaded")
    return _yolo_model


# ── Helpers ────────────────────────────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def is_user_logged_in(request):
    return request.session.get('user_logged_in', False)


def is_admin_logged_in(request):
    return request.session.get('admin_logged_in', False)


# ── Home ───────────────────────────────────────────────────────────────────────
def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})


# ── USER Registration ──────────────────────────────────────────────────────────
def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})
    if request.method == 'POST':
        username = request.POST.get('t1', '').strip()
        password = request.POST.get('t2', '').strip()
        confirm  = request.POST.get('t3', '').strip()
        email    = request.POST.get('t4', '').strip()
        fullname = request.POST.get('t5', '').strip()

        if not username or not password:
            return render(request, 'Register.html', {'data': 'Username and password are required.'})
        if password != confirm:
            return render(request, 'Register.html', {'data': 'Passwords do not match.'})
        if UserProfile.objects.filter(username=username).exists():
            return render(request, 'Register.html', {'data': 'Username already exists. Please choose another.'})

        user = UserProfile(
            username=username,
            password=hash_password(password),
            email=email,
            full_name=fullname,
        )
        user.save()
        return render(request, 'UserLogin.html', {'data': 'Registration successful! Please login.', 'success': True})


# ── USER Login ─────────────────────────────────────────────────────────────────
def UserLogin(request):
    if request.method == 'GET':
        if is_user_logged_in(request):
            return redirect('UserScreen')
        return render(request, 'UserLogin.html', {})


def UserLoginAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', '').strip()
        password = request.POST.get('t2', '').strip()
        hashed   = hash_password(password)

        try:
            user = UserProfile.objects.get(username=username, password=hashed)
            # Update last login
            user.last_login = timezone.now()
            user.save()
            # Store session
            request.session['user_logged_in'] = True
            request.session['username'] = username
            request.session['user_id']  = user.id
            return redirect('UserScreen')
        except UserProfile.DoesNotExist:
            return render(request, 'UserLogin.html', {'data': 'Invalid username or password.'})


def UserScreen(request):
    if not is_user_logged_in(request):
        return redirect('UserLogin')
    context = {'username': request.session.get('username', 'User')}
    return render(request, 'UserScreen.html', context)


# ── ADMIN Login ────────────────────────────────────────────────────────────────
def AdminLogin(request):
    if request.method == 'GET':
        if is_admin_logged_in(request):
            return redirect('AdminDashboard')
        return render(request, 'AdminLogin.html', {})


def AdminLoginAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', '').strip()
        password = request.POST.get('t2', '').strip()
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            request.session['admin_logged_in'] = True
            request.session['admin_username']  = username
            return redirect('AdminDashboard')
        else:
            return render(request, 'AdminLogin.html', {'data': 'Invalid admin credentials.'})


def AdminDashboard(request):
    if not is_admin_logged_in(request):
        return redirect('AdminLogin')
    users = UserProfile.objects.all().order_by('-date_joined')
    context = {
        'users': users,
        'total_users': users.count(),
        'active_users': users.filter(is_active=True).count(),
    }
    return render(request, 'AdminDashboard.html', context)


# ── Logout ─────────────────────────────────────────────────────────────────────
def Logout(request):
    request.session.flush()
    return redirect('index')


# ── FITNESS detection helpers ──────────────────────────────────────────────────
def getPlan(label):
    output = ""
    path = "FitnessApp/static/Guidance/" + label + ".txt"
    if os.path.exists(path):
        with open(path, "r") as file:
            for line in file:
                output += line.strip() + "<br/>"
    if len(output) > 0:
        data  = "<font size=4 color=blue>Exercise Plan Details &amp; Recommendations</font><br/>"
        data += "<font size=4 color=blue>Activity Recognized As : " + label + "</font><br/>"
        data  = data + output
        return data
    return ""


def detectActivity(frame):
    yolo_model = get_yolo_model()
    plan = ""
    labels = [
        'Bent Knee Crunch', 'Biceps Brachii Stretch', 'Bicycle Crunch', 'Boat Pose', 'BoundAnglePose',
        'Bridge', 'Camel Pose', 'Cat Cow Pose', 'Child Pose', 'Claim Exercise', 'Clam Exercise',
        'Cobra Pose', 'Cross Leg Forward Bend', 'Deltoid Muscle Stretch', 'Downward-Facing Dog Pose',
        'Frog Pose', 'Iliopsoas Muscle Stretch', 'InnerThighLift', 'LegRaise0', 'LegRaise10',
        'LegRaise30', 'LegRaise60', 'LegRaise90', 'Locust Pose', 'Lunge', 'Pigeon Pose', 'Plank',
        'Raise Leg Crunch', 'Reverse Prayer Pose', 'Seated Side Bend', 'SidePlank', 'Squat',
        'Standing Forward Bend', 'Super Men Pose', 'Triceps Stretch', 'Upper Trapezius Stretch'
    ]
    detections = yolo_model(frame)[0]
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        cls_id     = data[5]
        if float(confidence) >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, labels[int(cls_id)], (xmin, ymin + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            plan = getPlan(labels[int(cls_id)])
            break
    return frame, plan


# ── FITNESS views (protected) ──────────────────────────────────────────────────
def TrainDetection(request):
    if not is_user_logged_in(request) and not is_admin_logged_in(request):
        return redirect('UserLogin')
    if request.method == 'GET':
        img_b64  = None
        img_path = "model/result.png"
        if os.path.exists(img_path):
            cnn_train_detection = cv2.imread(img_path)
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(cnn_train_detection, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout(pad=0)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            plt.clf()
            plt.cla()
        context = {'data': 'CNN Activity Recognition Accuracy Graph', 'img': img_b64}
        return render(request, 'TrainDetection.html', context)


def Graph(request):
    if not is_user_logged_in(request) and not is_admin_logged_in(request):
        return redirect('UserLogin')
    if request.method == 'GET':
        img_b64  = None
        img_path = "model/results.png"
        if os.path.exists(img_path):
            cnn_train_graph = cv2.imread(img_path)
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(cnn_train_graph, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout(pad=0)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            plt.clf()
            plt.cla()
        context = {'data': 'CNN Training Graph', 'img': img_b64}
        return render(request, 'Graph.html', context)


def Predict(request):
    if not is_user_logged_in(request) and not is_admin_logged_in(request):
        return redirect('UserLogin')
    if request.method == 'GET':
        return render(request, 'Predict.html', {})


def PredictAction(request):
    if not is_user_logged_in(request) and not is_admin_logged_in(request):
        return redirect('UserLogin')
    if request.method == 'POST':
        filename = request.FILES['t1'].name
        image    = request.FILES['t1'].read()
        save_path = "FitnessApp/static/" + filename
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, "wb") as file:
            file.write(image)
        img = cv2.imread(save_path)
        img, plan = detectActivity(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (500, 300))
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context = {'data': plan, 'img': img_b64, 'username': request.session.get('username', '')}
        return render(request, 'UserScreen.html', context)


def PredictVideo(request):
    if not is_user_logged_in(request) and not is_admin_logged_in(request):
        return redirect('UserLogin')
    if request.method == 'GET':
        return render(request, 'PredictVideo.html', {})


def PredictVideoAction(request):
    if not is_user_logged_in(request) and not is_admin_logged_in(request):
        return redirect('UserLogin')
    if request.method == 'POST':
        filename = request.FILES['t1'].name
        image    = request.FILES['t1'].read()
        save_path = "FitnessApp/static/" + filename
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, "wb") as file:
            file.write(image)
        video_cap = cv2.VideoCapture(save_path)
        while True:
            ret, frame = video_cap.read()
            if ret:
                frame, plan = detectActivity(frame)
                cv2.imshow("Fitness Activity Output", frame)
                if cv2.waitKey(50) == ord("q"):
                    break
            else:
                break
        video_cap.release()
        cv2.destroyAllWindows()
        context = {'data': 'Processing Completed', 'username': request.session.get('username', '')}
        return render(request, 'UserScreen.html', context)
