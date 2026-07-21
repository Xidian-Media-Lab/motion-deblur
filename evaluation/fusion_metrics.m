function [SF, Qabf, VIF, EN, MI, FMI_pixel, FMI_dct, FMI_w] = fusion_metrics(image_f,image_nir,image_vis)
    grey_level = 256;
    [s1,s2] = size(image_nir);

    image1 = im2double(image_nir);
    image2 = im2double(image_vis);
    image_fused = im2double(image_f);
    
    %SF
    SF = SF_evaluation(image_fused);
    %Qabf
    Qabf = analysis_Qabf(image1, image2, image_fused);
    %VIF
    VIF = vifp_mscale(image_nir, image_f) + vifp_mscale(image_vis, image_f);
	%EN
    EN = entropy(image_f);
    %MI
    MI = MI_evaluation(image_nir,image_vis,image_f, grey_level);
    %FMI
    FMI_pixel = analysis_fmi(image1,image2,image_fused);
    FMI_dct = analysis_fmi(image1,image2,image_fused,'dct');
    FMI_w = analysis_fmi(image1,image2,image_fused,'wavelet');

end







