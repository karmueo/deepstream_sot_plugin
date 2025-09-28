#/bin/bash
/usr/src/tensorrt/bin/trtexec --loadEngine=sutrack_fp32.engine \
          --shapes=template:1x3x112x112,search:1x3x224x224,template_anno:1x4 \
          --iterations=100 \
          --percentile=95 \
          --exportProfile=profile.json \
          --separateProfileRun

# /usr/src/tensorrt/bin/trtexec --loadEngine=mixformer_v2_fp32.engine \
#           --shapes=img_t:1x3x112x112,img_ot:1x3x112x112,img_search:1x3x224x224 \
#           --iterations=100 \
#           --percentile=95 \
#           --exportProfile=profile.json \
#           --separateProfileRun