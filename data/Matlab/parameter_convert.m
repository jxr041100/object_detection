

id = importdata('nodeFeatureIndex.txt');
x  = importdata('featureRectX.txt');
y  = importdata('featureRectY.txt');
w  = importdata('featureRectW.txt');
h  = importdata('featureRectH.txt');


x = x(2:end);
y = y(2:end);
w = w(2:end);
h = h(2:end);

% id = [46, 13, 2, 84, 21, 106, 48, 47, 12, 26, 8, 96, 33, 57, 125, 53, 60, 7, 98, 44, 24, 42, 6, 61, 123, 9, 105, 30, 35, 119, 82, 52, 99, 18, 111, 92, 51, 28, 112, 69, 19, 73, 126, 45, 114, 97, 25, 34, 134, 16, 74, 113, 110, 37, 41, 117, 65, 56, 20, 15, 129, 102, 88, 59, 27, 29, 128, 70, 4, 76, 124, 54, 95, 39, 120, 89, 14, 67, 86, 80, 75, 55, 71, 31, 40, 109, 130, 66, 49, 133, 0, 103, 43, 90, 104, 85, 24, 78, 5, 11, 32, 122, 68, 58, 94, 87, 127, 3, 79, 116, 132, 62, 23, 77, 22, 72, 50, 108, 92, 81, 1, 121, 38, 17, 100, 63, 101, 118, 92, 10, 131, 83, 135, 115, 64, 36, 93, 91, 107];
% x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 18];
% y = [0, 0, 0, 1, 1, 1, 4, 11, 12, 14, 0, 1, 3, 7, 12, 13, 14, 17, 0, 1, 2, 3, 7, 11, 17, 0, 1, 7, 7, 18, 0, 3, 6, 6, 7, 8, 18, 0, 3, 6, 9, 10, 3, 4, 5, 5, 5, 6, 6, 18, 21, 0, 4, 9, 21, 0, 5, 5, 17, 18, 0, 2, 5, 11, 16, 16, 17, 18, 5, 5, 7, 8, 9, 10, 10, 14, 15, 15, 16, 16, 17, 21, 3, 5, 5, 5, 6, 7, 8, 10, 10, 15, 17, 18, 0, 1, 2, 3, 7, 10, 11, 12, 14, 17, 21, 6, 7, 11, 17, 17, 18, 18, 0, 1, 2, 7, 13, 13, 17, 0, 0, 4, 4, 6, 11, 13, 17, 17, 1, 3, 6, 16, 1, 1, 12, 0];
% w = [3, 4, 6, 2, 3, 3, 3, 3, 8, 4, 5, 2, 3, 4, 2, 4, 4, 3, 2, 2, 4, 4, 2, 2, 3, 2, 7, 2, 2, 2, 2, 2, 2, 2, 5, 4, 2, 2, 4, 2, 2, 2, 4, 4, 2, 2, 4, 4, 4, 1, 2, 3, 2, 5, 2, 1, 2, 3, 3, 1, 5, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 1, 2, 1, 3, 4, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 4, 1, 3, 3, 1, 3, 4, 1, 2, 2, 3, 3, 2, 3, 1, 2, 2, 1, 3, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3, 3, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2];
% h = [5, 2, 3, 3, 3, 7, 3, 4, 4, 3, 3, 2, 1, 4, 2, 1, 3, 2, 3, 2, 6, 4, 1, 3, 2, 2, 3, 1, 4, 2, 3, 1, 1, 5, 2, 3, 2, 2, 4, 5, 2, 2, 4, 3, 3, 5, 3, 2, 4, 2, 1, 7, 3, 1, 1, 4, 2, 2, 1, 2, 3, 6, 1, 1, 1, 1, 1, 1, 2, 3, 5, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 2, 2, 3, 6, 1, 2, 2, 1, 2, 1, 1, 1, 2, 5, 1, 3, 4, 2, 1, 2, 3, 1, 1, 5, 5, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 3, 2, 2, 3, 2, 6, 2, 4, 2, 2, 2, 3, 4, 1, 2, 2, 5, 2, 2];


idSize = id(1);
id = id(2:end);
ss = length(x);

idNew = -1 * ones(idSize,1);
xNew = -1 * ones(ss,1);
yNew = -1 * ones(ss,1);
wNew = -1 * ones(ss,1);
hNew = -1 * ones(ss,1);

marked =  -1 * ones(idSize,1);

id = id +1 ;

kk = 0;
for i=1:idSize
    if( marked(i) ~= 1)
        kk= kk +1;
        idT = find( id == id(i));
        if( marked(idT(1)) ~=1)
            marked(idT(1)) = 1;
            idNew(idT(1)) = kk;
            xNew(kk) = x(id(i));
            yNew(kk) = y(id(i));
            wNew(kk) = w(id(i));
            hNew(kk) = h(id(i));
        end
        if( length(idT) > 1)
            for j=2:length(idT)
                 marked(idT(j)) = 1;
                idNew(idT(j)) = kk;
            end
            
        end
    end
end
   
xNew = xNew';
yNew = yNew';
wNew = wNew';
hNew = hNew';
idNew = idNew';

 
id = idNew;
x = xNew;
y = yNew;
w = wNew;
h = hNew;

idNew = -1 * ones(idSize,1);
xNew = -1 * ones(idSize,1);
yNew = -1 * ones(idSize,1);
wNew = -1 * ones(idSize,1);
hNew = -1 * ones(idSize,1);

% marked =  -1 * ones(4788,1);

 
kk = 0;
for i=1:idSize
    
    kk= kk +1;
    idT = id(i);
    idNew(kk) = kk;
    xNew(kk) = x(id(i));
    yNew(kk) = y(id(i));
    wNew(kk) = w(id(i));
    hNew(kk) = h(id(i));
    
end

xNew = xNew';
yNew = yNew';
wNew = wNew';
hNew = hNew';
idNew = idNew';

idnew = idNew -1;

fid = fopen('featureRectX_1.txt','w');
for i=1:length(xNew)
    fprintf(fid, ' %d,',xNew(i));
end
fclose(fid);

fid = fopen('featureRectY_1.txt','w');
for i=1:length(yNew)
    fprintf(fid, ' %d,',yNew(i));
end
fclose(fid);

fid = fopen('featureRectW_1.txt','w');
for i=1:length(wNew)
    fprintf(fid, ' %d,',wNew(i));
end
fclose(fid);

fid = fopen('featureRectH_1.txt','w');
for i=1:length(hNew)
    fprintf(fid, ' %d,',hNew(i));
end
fclose(fid);




