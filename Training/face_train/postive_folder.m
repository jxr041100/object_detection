function imageNum = postive_folder(fid,sBasePath,subfolder,postive_folder)

image_name=sprintf('%s\\%s\\*.jpg',sBasePath,postive_folder);
list1=dir(image_name);
s=1;
e=length(list1);
imageNum = 0;
for i=s:e
    filename = list1(i).name;
    abs_filename=sprintf('%s\\%s\\%s',sBasePath,postive_folder,filename);
    image = imread(abs_filename);
    [height,width] = size(image);
    fprintf(fid,'%s\\%s\\%s %d %d %d %d %d\n',subfolder,postive_folder,filename,1,0,0,width-1,height-1);
    fprintf(1,'%s\\%s\\%s %d %d %d %d %d\n',subfolder,postive_folder,filename,1,0,0,width-1,height-1);
 imageNum = imageNum + 1;
end
fprintf(1, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\ntotal traffic sign detection samples: %d\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n',imageNum);
return
