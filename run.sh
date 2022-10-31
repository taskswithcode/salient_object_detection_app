option=${1?"Enter 1 for testing or 0 for production"}

if [ $option -eq 1 ]
then
    streamlit run app.py  --server.port 80 "1" "sod_app_examples.json" "sod_app_models.json"
else
    count=0
    while :
    do
        streamlit run app.py --server.port 8504  --server.enableCORS false --server.enableXsrfProtection false
        ((count=count+1))
        echo "Restarted $count times"
    done
fi

