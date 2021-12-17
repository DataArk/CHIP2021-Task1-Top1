if [[ $1 = 'medbert' ]];
then python ./medbert.py
elif [[ $1 = 'mcbert' ]];
then python ./mcbert.py
elif [[ $1 = 'macbert2-f-f' ]];
then python ./macbert2-f-f.py
elif [[ $1 = 'macbert2-f' ]];
then python ./macbert2-f.py
elif [[ $1 = 'dialog_chinese-macbert' ]];
then python ./dialog_chinese-macbert.py
elif [[ $1 = 'ensemble' ]];
then 
    python ./medbert.py
    python ./mcbert.py
    python ./macbert2-f-f.py
    python ./macbert2-f.py
    python ./dialog_chinese-macbert.py
    python ./ensemble.py
else python ./ensemble.py
fi
