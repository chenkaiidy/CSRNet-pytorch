from matplotlib import cm

from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

model = CSRNet()
model = model.cuda()
checkpoint = torch.load(sys.argv[1])
model.load_state_dict(checkpoint['state_dict'])

'''
img = F.to_tensor(Image.open(sys.argv[2]).convert('RGB'))
img = img/255.0
img[:,:,0]=(img[:,:,0]-0.485)/0.229
img[:,:,1]=(img[:,:,1]-0.456)/0.224
img[:,:,2]=(img[:,:,2]-0.406)/0.225
#img[0,:,:]=img[0,:,:]-92.8207477031
#img[1,:,:]=img[1,:,:]-95.2757037428
#img[2,:,:]=img[2,:,:]-104.877445883
img = img.cuda()
'''
img = transform(Image.open(sys.argv[2]).convert('RGB')).cuda()
output = model(img.unsqueeze(0))
output = output.detach().cpu().numpy()
print(output.sum())
plt.imshow(output , cmap = cm.jet )
plt.show()
