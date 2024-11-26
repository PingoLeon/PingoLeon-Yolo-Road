# Tasks

aim : train images of french road signs to create a detection model capable of making the difference between most of the signs, or at least the category which they belong to, like warning signs or order signs. Importance is made on the speed signs, most importants signs when driving and having to adapt vehicle's speed.

tweaks : dataset labeled and formed thanks to roboflow, labelIMG being too hard to understand. Compare Ultralytics YOLO differences betwwen latest v11 version and v? version, to se the differences and improvements in performance the model has made.

output : lightweight vision model (.pt, pytorch format) to run python scripts with it, such as giving it a video or anything else and be able to detect most of the signs visible (in a reasonable quality for the image / video)
