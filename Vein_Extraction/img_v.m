%% Processing image data
clc; clear; close all;
tic
% Select parameters for processing

mkdir('UP'); % Directory for saving processed vein_patterns
disp('Vein Extraction in PROGRESS.....');
%++++++
%%
% Please provide the path of IR and depth images
Files = dir('Codes\Vein_Extraction\Data\U10\*.png'); % copying files path

f = numel(Files); % No. of images in Folder titled 'Files'.
 Img_dp = imread('Codes\\Vein_Extraction\Data\U10\1_Depth.png'); % This refers to depth image
 Img_dp = rgb2gray(Img_dp);
Img_dp = imresize(Img_dp,[720 1280]);
%%
%for p = 1:2:f     % as we have two copies of each image (i.e. IR and  map)
for p = 2:1:f  % Ist iamge is depth map which is already read...

i =  ['Codes\Vein_Extraction\Data\U10\',Files(p).name]; %  Picking the image from the appropriate folder
Img_Ir = imread (i);  %===
%Img_Ir = rgb2gray(Img_Ir);
Img_Ir = imresize(Img_Ir,[720 1280]);
%Img_Ir = rgb2gray(Img_Ir);
%Img_Ir = adapthisteq(Img_Ir);% Adaptive histogram equilization
%h = fspecial('average',[5 5]); % filter
%Img_Ir=imfilter(Img_Ir,h,'replicate');



%%
% Setting ROI params

u1 = 690; u2= 950; v1 = 100; v2= 350;  % Set analytically

D = Img_dp(v1:v2,u1:u2); % Depth Image from ROI of the raw image
I = Img_Ir(v1:v2,u1:u2); % IR Image from ROI of the raw image

% Displaying Images
figure
fontSize = 15;
subplot(2, 2, 1);
imshow(Img_Ir);
title('Original IR Image', 'FontSize', fontSize);
subplot(2, 2, 2);
imshow(Img_dp);
title('Original Depth Map', 'FontSize', fontSize);
subplot(2, 2, 3);
imshow(I);
title(' IR Image {ROI}', 'FontSize', fontSize);
subplot(2, 2, 4);
imshow(D);
title(' Depth Map {ROI}', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'Position', get(0,'Screensize'));
%%
% Mask creation from D ...Thresholding
% Set masking parameter
z1 = 5 ; z2 = 35;  % Set analytically
[r,c] = size(D);
M = D; % Copy of depth map
for k = 1: r
    for l = 1:c
        
        if (D(k,l)>=z1 && D(k,l)<z2) % i.e., z1 and z2
           M(k,l)=1;
        else
           M(k,l)=0;
        end
    end
end
figure
imshow(M,[]); % Displaying image after thresolding
%%
% Scaling Down M. 
a =0.9; % Width Scaling Factor...These parameters helps in extracting the vein pattern (i.e., a=0.7, b= 0.8)... Other variations are already tested.
b =0.9; % Height scaling factor
nr = ceil(a*r); % New row --scaled down
nc = ceil(b*r); % New columns-- scaled down
N = imresize(M,[nr nc]); % Image resize using bicubic interpolation
figure
imshow(N,[]); % Displaying scaled down M, i.e., N.
% Cleaning N using 4- connected components --- this is done after
% extraction as well --- here 4 and 8 has a similar impact
CC = bwconncomp(N,4); % 4 Connected components
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
% Keeping only the largest and setting all others to 0
bgg = zeros(size(N));
bgg(CC.PixelIdxList{idx}) = 1; % Setting largets to 1, all others are already 0
figure
imshow(bgg,[]); % or simply imshow(bgg)
%%
% Extracting Vein pattern
% Resize I
Ir = imresize(I, [nr nc]);
[rr , cc] = size(Ir);
for i = 1:rr
    for j= 1:cc
        if bgg(i,j)==0
            Ir(i,j)=0;
        end
    end
end
%%
[Ox, X, Y,J] = stepblock(Ir,bgg);
figure
imshow(Ox)
%%
%rotate2 ; % Run rotate2 script --- analyzed using Radon Transform --- no
%significant imappct
%rotate;   % Does not have signiicant effect
%xx = bwmorph(Ox,'thin',1); % Line thining analyzed --- does not have
%significant affect
Ox = imresize(Ox,[231 231]);
filename = ['UP\test ',num2str(p),'.png'];
imwrite(Ox, filename); %  final processed image
close all;
end
disp('Vein Extraction COMPLETED....');
toc