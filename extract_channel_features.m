function brain_bands_power_values = extract_channel_features (channel_data, sampling_rate)

% This function is an example of feature extraction for EEG data. The
% features are the frequency power for 2 brain bands (delta and alpha)
%
% Each brain band correspond to a range of frequencies associated
% with different neural and cognitive functions (as defined for example
% here: https://en.wikipedia.org/wiki/Electroencephalography)
%

% Define brain bands frequency range (in Hz)
%delta_band = [1, 4];
theta_band = [4, 8];
alpha_band = [7.5, 13];
%beta_band = [13, 30];
%gamma_band = [30, 44];

% Compute frequency spectrum
[EEG_power, frequency_values, ~] = spect_analy(double(channel_data), ...
    length(channel_data), sampling_rate, 1);

% Find boundry positions for each band within the frequencies values array
% delta_s_freq = find(frequency_values >= delta_band(1));
% delta_e_freq = find(frequency_values <= delta_band(2));
% delta_start_freq = delta_s_freq(1);
% delta_end_freq = delta_e_freq(end);

theta_s_freq = find(frequency_values >= theta_band(1));
theta_e_freq = find(frequency_values <= theta_band(2));
theta_start_freq = theta_s_freq(1);
theta_end_freq = theta_e_freq(end);

alpha_s_freq = find(frequency_values >= alpha_band(1));
alpha_e_freq = find(frequency_values <= alpha_band(2));
alpha_start_freq = alpha_s_freq(1);
alpha_end_freq = alpha_e_freq(end);

% beta_s_freq = find(frequency_values >= beta_band(1));
% beta_e_freq = find(frequency_values <= beta_band(2));
% beta_start_freq = beta_s_freq(1);
% beta_end_freq = beta_e_freq(end);

% gamma_s_freq = find(frequency_values >= gamma_band(1));
% gamma_e_freq = find(frequency_values <= gamma_band(2));
% gamma_start_freq = gamma_s_freq(1);
% gamma_end_freq = gamma_e_freq(end);

%Extract power values for each band (feature value is just the addition)
% t_delta_m = EEG_power(:, delta_start_freq : delta_end_freq);
% power_delta = sum(t_delta_m);

t_theta_m = EEG_power(:, theta_start_freq : theta_end_freq);
power_theta = sum(t_theta_m);

t_alpha_m = EEG_power(:, alpha_start_freq : alpha_end_freq);
power_alpha = sum(t_alpha_m);

% t_beta_m = EEG_power(:, beta_start_freq : beta_end_freq);
% power_beta = sum(t_beta_m);

% t_gamma_m = EEG_power(:, gamma_start_freq : gamma_end_freq);
% power_gamma = sum(t_gamma_m);

% Return feature values
%brain_bands_power_values = [power_delta, power_theta, power_alpha, ...
%   power_beta, power_gamma];

brain_bands_power_values = [power_theta, power_alpha];


end


%% Included function 1
function [eegspecdB, freqs, specstd] = spect_analy( data, frames, srate, epoch_subset)

FREQFAC  = 2;  % approximate frequencies/Hz (default)
OVERLAP = 0;    % the overlap rate for pweltch

nchans = size(data,1);
%fftlength = 2^round(log(srate)/log(2))*g.freqfac;

winlength = max(pow2(nextpow2(frames)-3),4); %*2 since diveded by 2 later
winlength = min(winlength, 512);
winlength = max(winlength, 256);
winlength = min(winlength, frames);

fftlength = 2^(nextpow2(winlength))*FREQFAC;

%     usepwelch = 1;
usepwelch = license('checkout','Signal_Toolbox');
%     if ~license('checkout','Signal_Toolbox'),
% if ~usepwelch,
%     fprintf('\nSignal processing toolbox (SPT) absent: spectrum computed using the pwelch()\n');
%     fprintf('function from Octave which is suposedly 100%% compatible with the Matlab pwelch function\n');
% end;
% fprintf(' (window length %d; fft length: %d; overlap %d):\n', winlength, fftlength, g.overlap);

for c=1:nchans % scan channels or components

    tmpdata = data(c,:); % channel activity

    for e=epoch_subset
        if usepwelch
            [tmpspec,freqs] = pwelch(matsel(tmpdata,frames,0,1,e),...
                winlength,OVERLAP,fftlength,srate);
        else
            [tmpspec,freqs] = spec(matsel(tmpdata,frames,0,1,e),fftlength,srate,...
                winlength,OVERLAP);
        end;
        %[tmpspec,freqs] = psd(matsel(tmpdata,frames,0,1,e),fftlength,srate,...
        %					  winlength,g.overlap);
        if c==1 && e==epoch_subset(1)
            eegspec = zeros(nchans,length(freqs));
            specstd = zeros(nchans,length(freqs));
        end
        eegspec(c,:) = eegspec(c,:) + tmpspec';
        specstd(c,:) = specstd(c,:) + tmpspec'.^2;
    end
    % fprintf('.')
end

% log power of the data
eegspecdB = 10*log10( eegspec );

end

%% Included function 2
function [dataout] = matsel(data,frames,framelist,chanlist,epochlist)

if nargin<1
    help matsel
    return
end

[chans, framestot] = size(data);
if isempty(data)
    fprintf('matsel(): empty data matrix!?\n')
    help matsel
    return
end

if nargin < 5,
    epochlist = 0;
end
if nargin < 4,
    chanlist = 0;
end
if nargin < 3,
    fprintf('matsel(): needs at least 3 arguments.\n\n');
    return
end

if frames == 0,
    frames = framestot;
end

if framelist == 0,
    framelist = 1:frames;
end

framesout = length(framelist);

if isempty(chanlist) || chanlist == 0,
    chanlist = 1:chans;
end

chansout = length(chanlist);
epochs = floor(framestot/frames);

if epochs*frames ~= framestot
    fprintf('matsel(): data length %d was not a multiple of %d frames.\n',...
        framestot,frames);
    data = data(:,1:epochs*frames);
end

if isempty(epochlist) || epochlist == 0,
    epochlist = 1:epochs;
end
epochsout = length(epochlist);

if max(epochlist)>epochs
    fprintf('matsel() error: max index in epochlist (%d) > epochs in data (%d)\n',...
        max(epochlist),epochs);
    return
end

if max(framelist)>frames
    fprintf('matsel() error: max index in framelist (%d) > frames per epoch (%d)\n',...
        max(framelist),frames);
    return
end

if min(framelist)<1
    fprintf('matsel() error: framelist min (%d) < 1\n', min(framelist));
    return
end

if min(epochlist)<1
    fprintf('matsel() error: epochlist min (%d) < 1\n', min(epochlist));
    return
end

if max(chanlist)>chans
    fprintf('matsel() error: chanlist max (%d) > chans (%d)\n',...
        max(chanlist),chans);
    return
end

if min(chanlist)<1
    fprintf('matsel() error: chanlist min (%d) <1\n',...
        min(chanlist));
    return
end

dataout = zeros(chansout,framesout*epochsout);
for e=1:epochsout
    dataout(:,framesout*(e-1)+1:framesout*e) = ...
        data(chanlist,framelist+(epochlist(e)-1)*frames);
end
end
