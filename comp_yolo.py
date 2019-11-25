import yad2k
import patch
# get yolo bb
# get our binary classification map(bcm)

for b in bb:
    if bcm[b.x, b.y] > 0.5:
        out(b)
