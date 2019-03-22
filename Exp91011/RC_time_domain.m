function x_out = RC_time_domain (x_in,dt,f3dB)
% function x_out = RC_time_domain (x_in,dt,f3dB)
%
% simple RC filter arma mockup
% x_in is input data
% dt is sample rate (or could give full vector of timestamps)
% f3dB is 3dB cutoff frequency (or could give [R C]);
% 
% wjwiv 9 mar 2015

if length(dt) == length(x_in)
    % timestamps present, assume equally spaced
    dt = (dt(end) - dt(1))/(length(dt)-1);
    disp([' Assume equally spaced times, DT = ' num2str(dt) ' s']);
end

if length(f3dB) == 2
    R = f3dB(1); C = f3dB(2);
    tau = R*C; 
    disp([' Calculate tau = ' num2str(tau) ' s from R = ' ... 
        num2str(R) ' Ohm and C = ' num2str(C) ' F']);
elseif length(f3dB) ==1 
    tau = 1/(2*pi*f3dB);
end


output_decad = exp(-dt/tau);
innovazione = 1 - exp(-dt/tau); 

x_out =[0];
for jj = 1:length(x_in)-1
    x_jj_p1 = x_out(jj) * output_decad + innovazione * x_in(jj);
    x_out = [x_out; x_jj_p1];
end


