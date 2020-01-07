clear all; close all; clc
emotion='angry';
pts_folder=strcat('./C2GAN/datasets/RaFD_image_landmark/RaFD_landmark_pts/', emotion);
image_folder=strcat('./C2GAN/datasets/RaFD_image_landmark/RaFD_image/',emotion);
landmark_path=strcat('./C2GAN/datasets/RaFD_image_landmark/RaFD_landmark_map/', emotion);

pts_file=dir(pts_folder);
image_file=dir(image_folder);

if ~isdir(landmark_path)
    mkdir(landmark_path);
end


for k=3:length(pts_file)
    image_name = strcat(pts_file(k).name(1:length(pts_file(k).name)-10),'.jpg')
    image_path=fullfile(image_folder, image_name);
    pts_path=fullfile(pts_folder, pts_file(k).name);

    [FileId, errmsg]=fopen(pts_path);
    npoints=textscan(FileId,'%s %f',1,'HeaderLines',1);
    points=textscan(FileId,'%f %f',npoints{2},'MultipleDelimsAsOne',2,'Headerlines',2,'CollectOutput',1);
    fclose(FileId);
    point_data=points{1};

    I=imread(image_path);
    imshow(I);
    I(:,:,:)=255;
    imshow(I);

    hold on
    for i=1:length(point_data)
        final=plot(point_data(i,1),point_data(i,2), '-ko', 'MarkerSize', 4,'MarkerFaceColor','k');

        hold on
    end

    export_fig(fullfile(landmark_path,image_name));
    close all
end

    

