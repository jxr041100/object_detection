% get all positive images
sBasePath = 'C:\renj\TrainingData\speedlimit_positive\';
fid = fopen([sBasePath, '\GTDetectionSpeedBase.dat'],'w');

pos_foldername = 'Extracted_3rd_speed_limit';
subfolder = dir([sBasePath pos_foldername]);
e=length(subfolder);
num = 0;
for m = 3: e
    postivie_folder = subfolder(m).name;
    num = num+postive_folder(fid,[sBasePath pos_foldername],pos_foldername,postivie_folder);
end

fprintf(1,'the total number is %d\n',num);

pos_foldername = 'UCSD_LISA';
subfolder = dir([sBasePath pos_foldername]);
e=length(subfolder);
for m = 3: e
    postivie_folder = subfolder(m).name;
    num = num+postive_folder_png(fid,[sBasePath pos_foldername],pos_foldername,postivie_folder);
end

fprintf(1,'the total number is %d\n',num);
fclose(fid);






