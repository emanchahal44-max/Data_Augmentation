import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

input_image = r"C:\Users\SAAD COMPUTER\OneDrive\Desktop\task 4\OIP.jfif"  
save_folder = r"C:\Users\SAAD COMPUTER\OneDrive\Desktop\task 4\augmented_images"
os.makedirs(save_folder, exist_ok=True)  
augment = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomAffine(degrees=0, shear=20),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.5, 1.5))
])
img = Image.open(input_image)
num_augmented_images = 40
for i in range(num_augmented_images):
    augmented_img = augment(img)
    save_path = os.path.join(save_folder, f"image_aug_{i+1}.jpeg")
    augmented_img.save(save_path)
    plt.figure()
    plt.imshow(augmented_img)
    plt.axis('off')
    plt.show()
print(f" 40 augmented images saved in: {os.path.abspath(save_folder)}")
