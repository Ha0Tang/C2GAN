FILE=$1

if [[ $FILE != "RaFD_image_landmark" ]]; 
	then echo "Available datasets are RaFD_image_landmark"
	exit 1
fi


echo "Specified [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/datasets/C2GAN/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE