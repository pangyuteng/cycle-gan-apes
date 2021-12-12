#!/bin/bash

curl https://booksoficons.com/wp-content/uploads/2020/11/jason-calacanis.jpg --output static/jason-calacanis.jpg
curl https://i.pinimg.com/280x280_RS/56/81/1f/56811f7936a5187625b25893fe47fcbe.jpg --output static/chamath-palihapitiya.jpg
curl https://gossipgist.com/uploads/31116/david-friedberg-famous-for.png --output static/david-friedberg.png
curl https://global-uploads.webflow.com/5dfd5aca7badfa129f80056c/5f8790c5f9108d686c4bc35a_DavidSacksRecommendedBooks.jpeg --output static/david-sacks.jpg

CUDA_VISIBLE_DEVICES=1 python inference.py static/jason-calacanis.jpg static/ape-calacanis.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/david-friedberg.png static/ape-friedberg.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/chamath-palihapitiya.jpg static/ape-palihapitiya.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/david-sacks.jpg static/ape-sacks.png