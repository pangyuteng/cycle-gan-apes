#!/bin/bash

if [ ! -f static/jason-calacanis.jpg ]; then
    curl https://booksoficons.com/wp-content/uploads/2020/11/jason-calacanis.jpg --output static/jason-calacanis.jpg
    curl https://www.technologyspeakers.com/storage/celebrities/1538573772_chamath-palihapitiya.jpg --output static/chamath-palihapitiya.jpg
    curl https://gossipgist.com/uploads/31116/david-friedberg-famous-for.png --output static/david-friedberg.png
    curl https://global-uploads.webflow.com/5dfd5aca7badfa129f80056c/5f8790c5f9108d686c4bc35a_DavidSacksRecommendedBooks.jpeg --output static/david-sacks.jpg
fi

CUDA_VISIBLE_DEVICES=1 python inference.py static/jason-calacanis.jpg static/ape-calacanis.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/david-friedberg.png static/ape-friedberg.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/chamath-palihapitiya.jpg static/ape-palihapitiya.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/david-sacks.jpg static/ape-sacks.png
