function [O, D] = mua2OandD(mua, lambda)
%[O, D] = mua2OandD(dmua, lambda)
%   
%   Bigio, I., & Fantini, S. (2016). Quantitative Biomedical Optics:
%   Theory, Methods, and Applications (Cambridge Texts in Biomedical
%   Engineering). Cambridge: Cambridge University Press.
%   doi:10.1017/CBO9781139029797
%   
%   Inputs:
%       mua     - Time X wavelengths matrix of absorption data with in 
%                 1/cm.
%       lambda  - 1 X wavelengths vector of wavelengths in nm.
%   Outputs:
%       O       - Oxyhemoglobin concentration in uM.
%       D       - Deoxyhemoglobin concentration in uM.

    
    %% Parse Input
    if size(mua, 1)~=length(lambda)
        mua=mua.';
    end
    if size(lambda, 1)~=1
        lambda=lambda.';
    end
    
    %% Interpolate Extinction Spectra
    spectra=load('OandDspect.mat');
    Oext=interp1(spectra.lambda, spectra.Oext, lambda); %1/(mM cm)
    Dext=interp1(spectra.lambda, spectra.Dext, lambda); %1/(mM cm)
    
    %% Slove for O and D
    X=linsolve([Oext.', Dext.'], mua);
    O=X(1, :).'*1000; %uM
    D=X(2, :).'*1000; %uM
    
end

