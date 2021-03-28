function [p] = gaussian(x, mean, var)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
% mean = 0;
% var = 1;
p = 1/(sqrt(2*pi*var^2))*exp(-1*(x-mean).^2./(2*var^2));

end

