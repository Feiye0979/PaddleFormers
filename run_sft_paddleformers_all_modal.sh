#!/bin/sh

echo "Run sft multi-card text begin..."
sh run_sft_paddleformers_multi_card_text.sh > pdfms_omni_multi_card_text.log 2>&1
echo "Run sft multi-card text end..."

echo "Run sft multi-card image begin..."
sh run_sft_paddleformers_multi_card_image.sh > pdfms_omni_multi_card_image.log 2>&1
echo "Run sft multi-card image end..."

echo "Run sft multi-card audio begin..."
sh run_sft_paddleformers_multi_card_audio.sh > pdfms_omni_multi_card_audio.log 2>&1
echo "Run sft multi-card audio end..."

echo "Run sft multi-card omni begin..."
sh run_sft_paddleformers_multi_card_omni.sh > pdfms_omni_multi_card_omni.log 2>&1
echo "Run sft multi-card omni end..."

