import os
import shutil
import random

# Diretório raiz contendo as pastas de classes
dataset_root = 'frequenciasDefinitivas'

# Diretório de saída 
train_dir = 'train1'
val_dir = 'val1'
test_dir = 'test1'

# Proporção de imagens a serem usadas
train_ratio = 0.6
val_ratio = 0.2


labels = os.listdir(dataset_root)

for label in labels:
    label_dir = os.path.join(dataset_root, label)
    images = os.listdir(label_dir)
    random.shuffle(images)

    # Determina a quantidade de imagens para o treinamento, validação e teste
    num_train = int(train_ratio * len(images))
    num_val = int(val_ratio * len(images))

    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    for i, image in enumerate(images):
        if i < num_train:
            dest_dir = os.path.join(train_dir, label)
        elif i < num_train + num_val:
            dest_dir = os.path.join(val_dir, label)
        else:
            dest_dir = os.path.join(test_dir, label)
        shutil.copy(os.path.join(label_dir, image), os.path.join(dest_dir, image))

print("Divisão de treinamento, validação e teste concluída.")
