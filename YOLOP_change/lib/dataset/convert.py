id_dict = {
        0: "Pedestrian", 1: "Car", 2: "Bicycle", 3: "Truck", 4: "Bus", 5: "DangerCar", 6: "Van", 7: "Construction"
    }
    
id_dict_single = {0: "pedestrian"}

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
