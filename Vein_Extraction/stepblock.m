%% Function o for creating mean data
function [O,X, Y,J] = stepblock(X,Y)

 % Y is the matrix N from the main program
 % We need to find a mean value of block of oreder oxo, centered at (u,v)
  % X = resized I
  % Y = N
  K =18;
 [r1,c1]= size(X);
 J = zeros(r1,c1); % Matrix for holding vein pattern
 m = 0; % varaible for holding mean
  for i = 1:r1
      for j= 1:c1
          % finding mean of a block
          % Designing adaptive threshold filter
          % Setting Conditions for function O
          if i < ((K/2)+1) && j < ((K/2)+1)
               m = X(i:i+K,j:j+K);
              m = mean(m(:));
          elseif i > r1-((K/2)+1) && j > c1-((K/2)+1)
              m = X(i-K:i,j-K:j);
              m = mean(m(:));
          elseif i > r1-((K/2)+1) && j < c1-(K/2) && j > ((K/2)+1)
              m = X(i-K:i,j-(K/2):j+(K/2));
              m = mean(m(:));
          elseif j > c1-((K/2)+1) && i < r1-((K/2)+1) && i > ((K/2)+1)
              m = X(i-(K/2):i+(K/2),j-K:j);
              m = mean(m(:));
          elseif i > ((K/2)+1) && j < ((K/2)+1) && i <r1-((K/2)+1)
              m = X(i-(K/2):i+(K/2), j:j+K);
              m = mean(m(:));
          elseif i < ((K/2)+1) && j>((K/2)+1) && j<c1-((K/2)+1)
              m = X(i:i+K, j-(K/2):j+(K/2));
              m = mean(m(:));
          elseif i < ((K/2)+1) && j >c1-((K/2)+1)
              m = X(i:i+K, j-K:j);
              m = mean(m(:));
%           elseif i > r1-6 && j < 6
%               m = X(i-10:i, j:j+10);
%               m = mean(m(:));
           elseif i>((K/2)+1) && i<r1-((K/2)+1) && j>((K/2)+1) &&j<c1-((K/2)+1)
              m = X(i-(K/2):i+(K/2),j-(K/2):j+(K/2)); % Block of 5 helps in extracting vein pattern... other varitions are already tested
              m = mean(m(:));
              
          end
          % We define adative threshod fitler as;
          % J(i,j)= { 1; if X(i,j)> m and Y(i,j)==1;
          if (X(i,j)< m && Y(i,j)==1)
              J(i,j)=1;
          else
              J(i,j)=0;
          end
          m =0; % Setting m=0 for next iteration
          
      end
  end
  CC1 = bwconncomp(J,8);
  
  numPixels = cellfun(@numel, CC1.PixelIdxList);
  [biggest, idx] = max(numPixels);
  bgg = zeros(size(J));
  bgg(CC1.PixelIdxList{idx}) = 1;
  
 O = bgg; % return J
