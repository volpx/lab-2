% [fit_out,dfit_out,C,chi2,N_DOF]=fit_sine_poly(t,x,polyord,freqs,varargin)
%
% LSQ routine that fits data x(t) to a polynomial (order polyord) and any
% number of cos / sin terms with frequencies listed in freqs
% x(t) = a0 + a1*t + a2 * t^2 + ...
%           ... + C1 cos (2*pi*f1*t) + S1 sin (2*pi*f1*t)
%           ... + C2 cos (2*pi*f2*t) + S2 sin (2*pi*f2*t) + ...
%
% output:   fitout  (fit parameters): [a0; a1; ... a_polyord; C1; S1; C2; S2; ...]
%           dfit_out (fit uncertainties):   [da0; da1; ... ; dC1; dS1; ...]
%           C       full covariance matrix
%           chi2    fit chi^2 (meaningless if fit errors not given)
%           N_DOF   number of degrees of freedom
%
% options:
% 'err'     specify data uncertainties (next arg either a vector of errors
%           or a single number to be applied to all points ... otherwise
%           unity errors and a good fit are assumed)
% 't0'      next arg specifies time in which all sinusoids should start
%           (default t0=0)
% 'center'  fits polynomials to the function a_ordj*(t-t_c)^ordj,
%           where t_c is the mean of the in-range t values
% 'tran'    next arg is time range (fit only data in the time range)
% 'nopl', 'nobs' turn off the plotting and text summary outputs
%
% bw 14/12/2006

function [fit_out, dfit_out, C, chi2, N_DOF] = fit_sine_poly(t, x, polyord, freqs, varargin)
  
  plt  = 1;
  talk = 1;
  t0   = 0; % default: all sines have t0=0
  center = false;
  err_given = 0; % assume errors are NOT specified, and
  % thus we assign unity error to each point
  
  % Options checking
  if length(varargin) > 0
    for jj = 1:length(varargin)
      if strcmp(varargin{jj}, 'tran')
        tran = varargin{jj+1};
      end
      if strcmp(varargin{jj}, 'err')
        err = varargin{jj+1};
        err_given = 1;
      end
      if strcmp(varargin{jj}, 't0')
        t0 = varargin{jj+1};
      end
      if strcmp(varargin{jj}, 'nopl')
        plt = 0;
      end
      if strcmp(varargin{jj}, 'pl')
        plt = 1;
      end
      if strcmp(varargin{jj}, 'nobs')
        talk = 0;
      end
      if strcmp(varargin{jj}, 'talk')
        talk = 1;
      end
      if strcmp(varargin{jj}, 'center')
        center = true;
      end
    end
  end
  
  % Time range selection
  if exist('tran', 'var')
    in = (t >= tran(1) & t <= tran(2));
    tin = t(in);
    xin = x(in);
    if talk
      disp([' Fitting data between t = ' num2str(tran(1)) ' and ' num2str(tran(2)) ')']);
    end
  else
    tin = t;
    xin = x;
    if talk
      disp(['  Fitting all data (from t = ' num2str(min(t)) ' to ' num2str(max(t)) ')']);
    end
  end
  
  % Handling of uncertainties
  if ~exist('err', 'var')
    err = ones(length(tin), 1);
    if talk
      disp([' Assuming all points have unity error']);
    end
  elseif length(err) == 1
    if talk
      disp([' Assuming all points have error dx = ' num2str(err)]);
    end
    err = err * ones(length(tin), 1);
  else
    if talk
      disp([' Errors entered point by point']);
    end
    if length(tin) < length(err)
      err = err(in);
    end
  end
  
  if center
    t_c = mean(tin);
  else
    t_c = 0;
  end
  
  % Construct matrix of functions
  F = [];
  
  % Polynomial part
  for m = 0:polyord
    if talk
      disp([' Constructing polynomial function of order ' int2str(m) ]);
    end
    f = (tin - t_c).^m;
    F = [F f];
  end
  
  % Sinusoidal oscillation part
  if length(freqs) > 0
    for m = 1:length(freqs)
      F = [F, cos(2*pi*freqs(m)*(tin - t0)), sin(2*pi*freqs(m)*(tin - t0))];
    end
  end
  
  % Prepare matrix for the fit
  M = size(F, 2);
  G = [];
  V = []; % data vector
  for ii = 1:M
    V(ii) = sum (F(:, ii) .* xin ./ err.^2);
    for jj = 1:M
      G(ii,jj) = sum(F(:, ii) .* F(:, jj) ./ err.^2);
    end
  end
  V = V'; % it automatically makes V a row vector, we want a column vector
  
  % "function" matrix
  % "data" vector:
  
  C = inv(G);
  fit_out = C * V;
  
  x_fit = F * fit_out;
  dx_res = xin - x_fit;
  dx_mean = sum(dx_res) / (length(xin) - length(fit_out));
  
  % assign fit uncertainties, if necessary assuming a good fit (chi^2 = 1)
  N_DOF = length(xin) - length(fit_out);
  chi2 = sum(dx_res.^2 ./ err.^2) / N_DOF;
  
  if (err_given == 0)
    % here we assume a good fit, and scale the covariance matrix such that chi^2 would be = 1
    % this allows us to get an idea of the fit errors
    C = C * chi2;
  end
  dfit_out = sqrt(diag(C));
  
  sigma = sqrt(sum(dx_res.^2) / N_DOF);
  
  % Prepare a report
  if talk
    disp([' Fit model']);
    if t_c == 0
      polyarg = 't';
    else
      polyarg = '(t - t_c)';
    end
    
    for jj = 0:polyord
      if jj == 0
        coeff = ['a0'];
        term = coeff;
      else
        expon = int2str(jj);
        coeff = ['a' expon];
        term = [coeff ' * ' polyarg '^' expon];
      end
      if jj > 0
        model = [model ' + ' term];
      else
        model = [term];
      end
    end
    if length(freqs) > 0
      if t0 ~= 0
        sin_arg = '(t-t0)';
      else
        sin_arg = 't';
      end
      for jj = 1:length(freqs)
        coeffc = ['C' int2str(jj)];
        coeffs = ['S' int2str(jj)];
        fr = ['f' int2str(jj)];
        term = [coeffc ' cos 2*pi*' fr sin_arg ' + ' coeffs ' sin 2*pi*' fr sin_arg];
        model = [model ' + ' term];
      end
    end
    disp(['    ' model]);
    if center
      disp([' Using t_c = ' num2str(t_c)]);
    end
    if t0 ~= 0
      disp([' Using t0 = ' num2str(t0)]);
    end
    for ii = 1:length(freqs)
      disp([' f' int2str(ii) ': ' num2str(freqs(ii)) ' Hz']);
    end
    disp(['   ']);
    
    disp([' Fit results: ']);
    if err_given
      disp([' Fit chi^2 = ' num2str(chi2) ' per DOF (' int2str(N_DOF) ' DOF)']);
    else
      disp([' Assume good fit for fit parameter uncertainties']);
    end
    disp([' Average fit residual = ' num2str(dx_mean) ]);
    disp([' RMS residual = ' num2str(sigma)]);
    disp([' Fit parameters: ']);
    if polyord >= 0
      for ii = 0:polyord
        disp([' Polynomial order ' int2str(ii) ': ' num2str(fit_out(ii+1)) ' +/- ' ...
          num2str(dfit_out(ii+1))]);
      end
    end
    if length(freqs) > 0
      for ii = 1:length(freqs)
        disp([' Frequency ' num2str(freqs(ii)) ' Hz, cosine: ' num2str(fit_out(polyord + 1 + (ii-1)*2 + 1)) ' +/- ' ...
          num2str(dfit_out(polyord + 1 + (ii-1)*2 + 1)) ', sine: ' num2str(fit_out(polyord + 1 +(ii-1)*2 + 2)) ' +/- ' ...
          num2str(dfit_out(polyord + 1 + (ii-1)*2 + 2)) ]);
      end
    end
    disp(['   ']);
  end
  
  % A plot
  if plt
    figure
    axtop = axes('position', [.15 .5 .75 .4]);
    plot (tin, xin, 'b');
    hold on
    plot (tin, x_fit, 'r');
    grid on;
    axbot = axes('position', [.15 .1 .75 .3]);
    plot (tin, dx_res, 'r');
    grid on;
  end
  
end
