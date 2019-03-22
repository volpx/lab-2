function x_out = RC_double_time_domain (x_in,dt,components)
% function x_out = RC_double_time_domain (x_in,dt,components)
%
% simple double RC filter arma mockup
% x_in is input data
% dt is sample rate (or could give full vector of timestamps)
% components are the RC values ([R1 C1 R2 C2])
% 
% wjwiv 9 mar 2015

if length(dt) == length(x_in)
    % timestamps present, assume equally spaced
    dt = (dt(end) - dt(1))/(length(dt)-1);
    disp([' Assume equally spaced times, DT = ' num2str(dt) ' s']);
end

R1 = components(1);
C1 = components(2);
R2 = components(3);
C2 = components(4); 

tau_1 = R1*C1;
tau_2 = R2*C2;
alpha = R1 / R2; 

mid_decad = exp(-dt/tau_1*(1 + alpha));
mid_in_innovazione = 1 / (1 + alpha) * (1 - exp(-dt/tau_1*(1 + alpha)));
mid_out_innovazione = alpha / (1 + alpha) * (1 - exp(-dt/tau_1*(1 + alpha)));; 

out_decad = exp(-dt/tau_2);
out_mid_innovazione = 1 - exp(-dt/tau_2); 

x_out = [0];
x_mid = 0;
for jj = 1:length(x_in)-1
    x_jj_p1_out = x_out(jj) * out_decad + out_mid_innovazione * x_mid;
    x_mid = x_out(end) * mid_out_innovazione + x_in(jj) * mid_in_innovazione ... 
        + x_mid * mid_decad;     
    x_out = [x_out; x_jj_p1_out];
end
