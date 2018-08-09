
# CONFIGURATION
trainFile='../TrainTestCorpus/mediumTrain1.csv'
ngram=2
kFeat=10000
stopWords='english'
interval=1000
limit=100
negTag=True
binary=False
update=False
bernoulli=False
# HOST & ACCESS
host='index-fr.semantiweb.fr'
user='chuyuan'
pw='chuyuan'
db='ratings_and_reviews_ml'
table='customers_avis_smell'

# TRIGGER THE SCRIPT
touch trace.txt
echo "" >> trace.txt
echo "Lancer le script nbPredit.py ..."
python3 nbPredit.py $trainFile -n=$ngram -k=$kFeat -nt=$negTag -s=$stopWords -i=$interval -l=$limit -b=$binary -up=$update -ber=$bernoulli -ho=$host -u=$user -pw=$pw -db=$db -t=$table>> trace.txt
echo "End of execution nbPredit.py"

