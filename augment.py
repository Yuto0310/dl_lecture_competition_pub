import os
import json
from PIL import Image
from torchvision import transforms
import pandas as pd
from multiprocessing import Pool, cpu_count

# データ拡張の定義
augmentations = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10), 
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
])

# Resizeの定義
resize_transform = transforms.Resize((224, 224))

def process_image(args):
    index, row, image_dir, save_dir, num_augmentations = args
    image_path = os.path.join(image_dir, row['image'])
    image = Image.open(image_path)

    # 元の画像をリサイズして保存
    resized_image = resize_transform(image)
    original_image_path = os.path.join(save_dir, f'{index}_original.png')
    resized_image.save(original_image_path)

    new_data = [{
        'image': f'{index}_original.png',
        'question': row['question'],
        'answers': row.get('answers', None)
    }]

    # 拡張画像を保存
    for j in range(num_augmentations):
        augmented_image = augmentations(image)
        augmented_image = transforms.ToPILImage()(augmented_image)
        resized_augmented_image = resize_transform(augmented_image)
        augmented_image_index = f"{index}_{j}"
        augmented_image_path = os.path.join(save_dir, f'{augmented_image_index}.png')
        resized_augmented_image.save(augmented_image_path)

        new_data.append({
            'image': f'{augmented_image_index}.png',
            'question': row['question'],
            'answers': row.get('answers', None)
        })

    return new_data

def augment_and_save(df_path, image_dir, save_dir, num_augmentations=2):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_json(df_path, orient='columns')

    # マルチプロセシングの設定
    num_processes = cpu_count()
    pool = Pool(processes=num_processes)

    args = [(i, row, image_dir, save_dir, num_augmentations) for i, row in df.iterrows()]
    results = pool.map(process_image, args)

    # 結果を結合してDataFrameに変換
    new_data = [item for sublist in results for item in sublist]
    augmented_df = pd.DataFrame(new_data)
    augmented_df.to_json(os.path.join(save_dir, 'augmented_train.json'), orient='columns')

if __name__ == "__main__":
    augment_and_save(df_path="./data/train.json", image_dir="./data/train", save_dir="./data/augmented_train")
