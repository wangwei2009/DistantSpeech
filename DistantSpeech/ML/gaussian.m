function [p] = gaussian(x, mean, var)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
% mean = 0;
% var = 1;
p = 1/(sqrt(2*pi*var^2))*exp(-1*(x-mean).^2./(2*var^2));

end

