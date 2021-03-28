
close all


mean = 5;
var = 1;
p = gaussian(0,mean,var)
fun = @(x)1/(sqrt(2*pi*var^2))*exp(-1*(x-mean).^2./(2*var^2));

q = integral(fun,-15,15)

n = -15:1:15;
p = zeros(length(n),1);
cdf = zeros(length(n),1);
s = 0;
for i = 1:length(n)
    p(i) = gaussian(n(i),mean,var);
    s = s + p(i);
    cdf(i) = s;
end
figure,plot(n, p)
figure,plot(n,cdf)
