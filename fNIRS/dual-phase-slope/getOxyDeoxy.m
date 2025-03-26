function [ I, phi, O_DSI, D_DSI, O_DSphi, D_DSphi] = getOxyDeoxy(A_I, B_I, A_phi, B_phi, t, ...
                                                                 A_I_all, B_I_all, A_phi_all, B_phi_all, t_all, ...
                                                                 lambda) 

    % Uses Giles Blaney's example to run Dual Phase Slope calculations and return the results
        
    I(:, :, 1)=[A_I(:, 1), B_I(:, 1), B_I(:, 2), A_I(:, 2)]; %counts
    I(:, :, 2)=[A_I(:, 3), B_I(:, 3), B_I(:, 4), A_I(:, 4)]; %counts

    phi(:, :, 1)=[A_phi(:, 1), B_phi(:, 1), B_phi(:, 2), A_phi(:, 2)] * pi/180; %rad
    phi(:, :, 2)=[A_phi(:, 3), B_phi(:, 3), B_phi(:, 4), A_phi(:, 4)] * pi/180; %rad    


    % use all of a participant's data for the baseline calculations
    I_all(:, :, 1)=[A_I_all(:, 1), B_I_all(:, 1), B_I_all(:, 2), A_I_all(:, 2)]; %counts
    I_all(:, :, 2)=[A_I_all(:, 3), B_I_all(:, 3), B_I_all(:, 4), A_I_all(:, 4)]; %counts

    phi_all(:, :, 1)=[A_phi_all(:, 1), B_phi_all(:, 1), B_phi_all(:, 2), A_phi_all(:, 2)] * pi/180; %rad
    phi_all(:, :, 2)=[A_phi_all(:, 3), B_phi_all(:, 3), B_phi_all(:, 4), A_phi_all(:, 4)] * pi/180; %rad


    %% Calculate O and D with Dual-Slope (DS)
    opts.rho=[25, 35]; %mm
    opts.nin=1.333;
    opts.fmod=110e6; % changed per conv. with Giles B. 11/2/2023 from 140.625e6 for HCILAB fNIRS device; %Hz
    opts.mua=0.01; %1/mm
    opts.musp=1; %1/mm

    % use all of a participant's data for the baseline calculations
    opts.blInds=1:length(t_all);    

    initVar=NaN(length(t), size(I, 3));
    dmua_DSI=initVar;
    dmua_DSphi=initVar;
    clear initVar;
    for lInd=1:size(I, 3)
        dmua_DSI(:, lInd)=DSdmua(I(:, :, lInd), I_all(:, :, lInd), 'intensity', opts); %1/mm
        dmua_DSphi(:, lInd)=DSdmua(phi(:, :, lInd), phi_all(:, :, lInd), 'phase', opts); %1/mm
    end

    [O_DSI, D_DSI]=mua2OandD(dmua_DSI*10, lambda); %uM
    [O_DSphi, D_DSphi]=mua2OandD(dmua_DSphi*10, lambda); %uM

end