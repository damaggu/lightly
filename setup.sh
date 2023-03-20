#git clone https://github.com/damaggu/lightly.git
#cd lightly
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
pip install -r requirements/video.txt
pip install -r requirements/mine

mkdir datasets
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xvzf imagenette2-160.tgz
mv imagenette2-160 datasets/imagenette2-160
rm imagenette2-160.tgz

# from https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/?p=%2Fckpts&mode=list
mkdir vqgan
cd vqgan
git clone https://github.com/CompVis/taming-transformers.git
mv taming-transformers taming_transformers
mv taming_transformers vqgan/
#wget --header 'Host: heibox.uni-heidelberg.de' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/?p=%2Fconfigs&mode=list' --header 'Cookie: sfcsrftoken=wzoShoBSab0RXNy96zdWaj4XuAndrPiR0eWe7tfFT5uFcJoQf96BRCz2DQz8eIp4' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: same-origin' --header 'Sec-Fetch-User: ?1' 'https://heibox.uni-heidelberg.de/seafhttp/files/e889381a-3680-457a-81e0-f7be06dea036/model.yaml' --output-document 'model.yaml'
#wget --header 'Host: heibox.uni-heidelberg.de' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/?p=%2Fckpts&mode=list' --header 'Cookie: sfcsrftoken=wzoShoBSab0RXNy96zdWaj4XuAndrPiR0eWe7tfFT5uFcJoQf96BRCz2DQz8eIp4' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: same-origin' --header 'Sec-Fetch-User: ?1' 'https://heibox.uni-heidelberg.de/seafhttp/files/be2cce3e-6fcf-4d5e-913b-6ac3c29eb281/last.ckpt' --output-document 'last.ckpt'