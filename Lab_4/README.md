# Fruit Object Detection — DL Lab 4

A lab that teaches object detection by building it up step by step — from scratch math all the way to training a real YOLO model that can spot apples, bananas, and oranges in photos.

---

## What's the goal?

We want a model that can look at a photo and draw boxes around fruits, labeling each one. To get there, the lab walks through how researchers solved this problem over the years — each task fixes a problem from the previous one.

---

## The Dataset

Photos of fruits (apples, bananas, oranges) with hand-labeled bounding boxes stored as XML files. One XML file per image tells us where each fruit is and what it's called.

---

## What we build, step by step

### Step 0 — Look at the data first
Before doing anything fancy, we just draw the labeled boxes on random images to make sure we understand what the data looks like.

---

### Step 1 — IoU (How do we know if a prediction is good?)

**The problem:** If a model predicts a box around a banana, how do we score it? Is it close enough to the real box?

**The answer:** IoU — Intersection over Union. We measure how much the predicted box overlaps with the real box. 1.0 means perfect, 0.0 means no overlap at all.

```
IoU = overlapping area / total combined area
```

We implement this from scratch and test it on three cases: big overlap, small overlap, and no overlap at all.

---

### Step 2 — Selective Search (Where might objects be?)

**The problem:** The model needs to know *where to look* before it can classify anything.

**The answer:** Selective Search — an algorithm that looks at colors and textures to suggest ~2,000 regions in the image that might contain an object. Think of it as the model's "attention" before attention was a thing.

We generate 200 proposals and draw them all on the image — it looks like a mess of rectangles, but most real objects are covered somewhere in there.

---

### Step 3 — R-CNN (The slow original)

**The idea:** Take each of those 200 proposals, crop that region out of the image, and run it through a neural network (ResNet-18) to get a feature description.

**The problem we discover:** Doing 200 separate crops × 200 separate neural network passes is painfully slow. On a CPU, 100 crops alone takes several seconds. For a full 2,000-proposal run, you're looking at minutes per image. Not usable in the real world.

This task exists to *feel* the bottleneck, not to ship a product.

---

### Step 4 — Fast R-CNN (The smart fix)

**The insight:** Why run the neural network 200 times? Run it once on the whole image, then just look up the features for each region.

That's exactly what Fast R-CNN does. One forward pass produces a feature map of the whole image. Then a technique called **RoI Pooling** crops the relevant part of that feature map for each proposal — all in one shot.

**Result:** Same 100 proposals, but now it's ~10–50× faster because the expensive computation only happens once.

---

### Step 5 — Faster R-CNN (The end-to-end version)

**The remaining problem:** We're still using Selective Search (slow, not learned) to generate proposals.

**The fix:** Replace Selective Search with a small neural network called an RPN (Region Proposal Network) that runs inside the same model. Now everything is learned together — proposals, features, and classification — in one pipeline.

We use a pretrained Faster R-CNN from PyTorch trained on 80 COCO categories. It already knows what bananas and apples look like. We run it, filter out low-confidence predictions, and visualize the results.

---

### Step 6 — NMS (Cleaning up duplicate boxes)

**The problem:** Even a good model will predict 5 boxes around the same apple. We need to pick just one.

**The fix:** Non-Maximum Suppression. Sort all predictions by confidence. Keep the best one. Throw away any other box that overlaps too much with it. Repeat.

We implement this from scratch and see it collapse 5 overlapping boxes down to 2 clean ones.

---

### Step 7 — YOLOv8 (Fast, modern, fine-tuned)

**The big idea shift:** All the R-CNN models are *two-stage* — first propose regions, then classify them. YOLO throws that away and does everything in a single pass through the network.

The result: instead of hundreds of milliseconds, YOLO runs in ~20–50ms. That's fast enough for video.

**What we do:**
1. Convert the dataset from XML format to the format YOLO expects
2. Fine-tune the smallest YOLO model (`yolov8n`) on our 3-class fruit dataset for 10 epochs
3. Evaluate it with standard metrics (precision, recall, mAP)
4. Compare it side-by-side with Faster R-CNN

---

## How the models compare

| Model | Speed | Notes |
|---|---|---|
| R-CNN | Very slow | Runs CNN once per region — educational only |
| Fast R-CNN | Faster | Shared features, still needs Selective Search |
| Faster R-CNN | ~200–400ms | Fully learned, good accuracy |
| YOLOv8 (fine-tuned) | ~20–50ms | Best speed, trained on our fruits specifically |

---

## Files

```
dl-lab-4.ipynb     the full notebook
fruit.yaml         dataset config for YOLO
fruit_dataset/     images + converted labels (created during training)
runs/              YOLO training outputs and best model weights
```

---

## How to run it

This notebook is designed for **Kaggle** (the dataset paths point to Kaggle's input directory). To run it:

1. Upload to Kaggle and add the `fruit-images-for-object-detection` dataset
2. Run cells top to bottom
3. GPU is recommended for Tasks 5 and 7

---

## What you'll learn

By the end of this lab you'll understand why modern detectors look the way they do — every design choice (RoI Pooling, RPN, single-pass detection) exists because the previous approach had a specific, measurable problem. The lab makes you feel each bottleneck before showing the fix.
