import torch
import torch.utils.data
from torchvision import transforms, datasets
import os
import random
import shutil

def clarify(ratel, start, end):#lsize is [a,b)
    bigL=[]
    numbers = list(range(start, end))
    #print(numbers)
    total_num=len(numbers)
    
    # Allocate numbers to l1, l2, l3 based on rates，最后一个或为空
    for i, rate in enumerate(ratel):
        allocation_count = int(total_num * rate)
        if(i==len(ratel)-1):
          allocation_count=total_num
          for tmpi in range(0,i):
            allocation_count=allocation_count-len(bigL[tmpi])
        if allocation_count > 0:
          selected_numbers = random.sample(numbers, k=allocation_count)
          numbers = [num for num in numbers if num not in selected_numbers]
          bigL.append(selected_numbers)
        elif allocation_count == 0:
          bigL.append([])
    # Ensure all numbers are allocated
    assert len(numbers) == 0, "Allocation failed, numbers not fully allocated"
    return bigL
  
def main():
  ratel=[0.7,0.15,0.15]
  namel=["train","val","test"]
  all_dataset = datasets.ImageFolder('./NCT-CRC-HE-100K/NCT-CRC-HE-100K')
  seed=0
  output_dir = './data'
  print("****")
  random.seed(seed)
  print(len(all_dataset))
  print(all_dataset.classes)
  print(all_dataset.class_to_idx)
  print(f"开始划分{len(all_dataset)},{namel},{ratel}")
  
  pre=0
  lsize=[]
  prei=0#记录每个类别的第一个
  for i, image in enumerate(all_dataset.imgs):
    if(pre!=image[1]):
      lsize.append(i-prei)
      pre=image[1]
      prei=i
    elif(i==len(all_dataset.imgs)-1):
      lsize.append(i-prei+1)
  print(lsize)#lsize存储各个类别的数量
  #return
  # 创建输出文件夹
  
  name_type_dir=[]
  for name in namel:
    name_type_dir.append(os.path.join(output_dir,name))
  print("name_type_dir",name_type_dir)
  # 创建类别文件夹
  for dir in name_type_dir:
      if not os.path.exists(dir):
          os.makedirs(dir)
      for typedir in all_dataset.classes:
        typedir = os.path.join(dir,typedir)
        if not os.path.exists(typedir):
          os.makedirs(typedir)
  #循环遍历每个类别
  
  start = 0
  end = 0
  fault = 0
  for i , num in enumerate(lsize):
    start=end
    end=start+num
    print(f"start{start},end{end}")
    bigL=clarify(ratel, start, end)
    print("***bigL*print**")
    for l in bigL:
      print(l)
    #j是整个数据集的下标，j下标属于l1分给train，下标为l2分给test，l3分给valid
    for j in range(start,end):
      print("***a***")
      image_path=all_dataset.imgs[j][0]
      print("image_path:{}".format(image_path))
      typedir=all_dataset.classes[all_dataset.imgs[j][1]]
      print("typedir:{}".format(typedir))
      print("j:{}".format(j))
      fault=1
      for idx,l in enumerate(bigL):
        if(j in l):
          label_dir=os.path.join(name_type_dir[idx],typedir)
          print("label_dir:{0}".format(label_dir))
          shutil.copy(image_path, label_dir)
          fault=0
      if(fault):
        print("wrong!!,not alloced")
    print("***b***")
  '''
  train_dataset = datasets.ImageFolder('./dataset/train')
  test_dataset = datasets.ImageFolder('./dataset/test')
  valid_dataset = datasets.ImageFolder('./dataset/val')
  print(f"统计train:{len(train_dataset)},test:{len(test_dataset)},valid:{len(valid_dataset)}")
  '''
  print("***开始统计***")
  for idx,path in enumerate(name_type_dir):
    dataset=datasets.ImageFolder(path)
    print(f"type:{namel[idx]};len:{len(dataset)}")
  print("***统计结束***")
if __name__ == "__main__":
    main()
