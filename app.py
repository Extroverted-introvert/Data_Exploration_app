from  flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import psycopg2
import pandas as pd
import numpy as np
from sklearn import svm
from Linear_reg import Linear_Regression
from Non_Linear import Non_Linear_Regression
from Logistic_Reg import Logistic_Regression
from Decision_Tree import Decision_Tree
from Random_forest import Random_forest
from SVM import SVM_model
from pca import PCA
from KNN import KNN
from Kmeans import Kmeans
from DBSCAN import DBSCAN
from Naive_Bayes import Naive_Bayes
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import scale

#from flask_sqlalchemy import SQLAlchemy
app=Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI']='postgres://xmswjltwtgrzhm:73c8e46bdf06302bc30ede273c58f22a9cc3280cb387a5146c08f9d42512114d@ec2-34-193-232-231.compute-1.amazonaws.com:5432/d3q62fgfljcc4a'
app.config['SQLALCHEMY_DATABASE_URI']='postgres://postgres:parth@123@localhost/postgres'
db = SQLAlchemy(app)


class Linear_table(db.Model):
    __tablename__='Linear'
    id=db.Column(db.Integer, primary_key=True)
    engine=db.Column(db.Float,nullable=False)
    co2=db.Column(db.Float,nullable=False)
    
    def _init__(id,engine,co2):
        self.id=id
        self.engine=engine
        self.co2=co2

class Non_linear_table(db.Model):
    __tablename__='Non_linear'
    id=db.Column(db.Integer, primary_key=True)
    year=db.Column(db.Float,nullable=False)
    gdp=db.Column(db.Float,nullable=False)
    
    def _init__(id,year,gdp):
        self.id=id
        self.year=year
        self.gdp=gdp



class Logistic_table(db.Model):
    __tablename__='Logistic'
    id=db.Column(db.Integer, primary_key=True)
    tenure=db.Column(db.Float,nullable=False)
    age=db.Column(db.Float,nullable=False)
    address=db.Column(db.Float,nullable=False)
    income=db.Column(db.Float,nullable=False)
    education=db.Column(db.Float,nullable=False)
    employ=db.Column(db.Float,nullable=False)
    equip=db.Column(db.Float,nullable=False)
    churn=db.Column(db.Float,nullable=False)
        
    def _init__(id,tenure,age,address,income,education,employ,equip,churn):
        self.id=id
        self.tenure=tenure
        self.age=age
        self.address=address
        self.income=income
        self.education=education
        self.employ=employ
        self.equip=equip
        self.churn=churn

class Decision_table(db.Model):
    __tablename__='Decision'
    id=db.Column(db.Integer, primary_key=True)
    age=db.Column(db.Float,nullable=False)
    sex=db.Column(db.Float,nullable=False)
    bp=db.Column(db.Float,nullable=False)
    cholestrol=db.Column(db.Float,nullable=False)
    na_to_k=db.Column(db.Float,nullable=False)
    drug=db.Column(db.String,nullable=False)
        
    def _init__(id,age,sex,bp,cholestrol,na_to_k,drug):
        self.id=id
        self.age=age
        self.sex=sex
        self.bp=bp
        self.cholestrol=cholestrol
        self.na_to_k=na_to_k
        self.drug=drug


class SVM_table(db.Model):
    __tablename__='SVM'
    id=db.Column(db.Integer, primary_key=True)
    clump=db.Column(db.Float,nullable=False)
    unifsize=db.Column(db.Float,nullable=False)
    unifshape=db.Column(db.Float,nullable=False)
    margadh=db.Column(db.Float,nullable=False)
    singepisize=db.Column(db.Float,nullable=False)
    barenuc=db.Column(db.Float,nullable=False)
    blandchrom=db.Column(db.Float,nullable=False)
    normnucl=db.Column(db.Float,nullable=False)
    mit=db.Column(db.Float,nullable=False)
    cell_class=db.Column(db.Float,nullable=False)
        
    def _init__(id,clump,unifsize,unifshape,margadh,singepisize,barenuc,blandchrom,normnucl,mit,cell_class):
        self.id=id
        self.clump=clump
        self.unifsize=unifsize
        self.unifshape=unifshape
        self.margadh=margadh
        self.singepisize=singepisize
        self.barenuc=barenuc
        self.blandchrom=blandchrom
        self.normnucl=normnucl
        self.mit=mit
        self.cell_class=cell_class

class Random_forest_table(db.Model):
    __tablename__='Random_forest'
    id=db.Column(db.Integer, primary_key=True)
    clump=db.Column(db.Float,nullable=False)
    unifsize=db.Column(db.Float,nullable=False)
    unifshape=db.Column(db.Float,nullable=False)
    margadh=db.Column(db.Float,nullable=False)
    singepisize=db.Column(db.Float,nullable=False)
    barenuc=db.Column(db.Float,nullable=False)
    blandchrom=db.Column(db.Float,nullable=False)
    normnucl=db.Column(db.Float,nullable=False)
    mit=db.Column(db.Float,nullable=False)
    cell_class=db.Column(db.Float,nullable=False)
        
    def _init__(id,clump,unifsize,unifshape,margadh,singepisize,barenuc,blandchrom,normnucl,mit,cell_class):
        self.id=id
        self.clump=clump
        self.unifsize=unifsize
        self.unifshape=unifshape
        self.margadh=margadh
        self.singepisize=singepisize
        self.barenuc=barenuc
        self.blandchrom=blandchrom
        self.normnucl=normnucl
        self.mit=mit
        self.cell_class=cell_class

class PCA_table(db.Model):
    __tablename__='PCA'
    id=db.Column(db.Integer, primary_key=True)
    clump=db.Column(db.Float,nullable=False)
    unifsize=db.Column(db.Float,nullable=False)
    unifshape=db.Column(db.Float,nullable=False)
    margadh=db.Column(db.Float,nullable=False)
    singepisize=db.Column(db.Float,nullable=False)
    barenuc=db.Column(db.Float,nullable=False)
    blandchrom=db.Column(db.Float,nullable=False)
    normnucl=db.Column(db.Float,nullable=False)
    mit=db.Column(db.Float,nullable=False)
    cell_class=db.Column(db.Float,nullable=False)
        
    def _init__(id,clump,unifsize,unifshape,margadh,singepisize,barenuc,blandchrom,normnucl,mit,cell_class):
        self.id=id
        self.clump=clump
        self.unifsize=unifsize
        self.unifshape=unifshape
        self.margadh=margadh
        self.singepisize=singepisize
        self.barenuc=barenuc
        self.blandchrom=blandchrom
        self.normnucl=normnucl
        self.mit=mit
        self.cell_class=cell_class

class KNN_table(db.Model):
    __tablename__='KNN'
    id=db.Column(db.Integer, primary_key=True)
    region=db.Column(db.Float,nullable=False)
    tenure=db.Column(db.Float,nullable=False)
    age=db.Column(db.Float,nullable=False)
    marital=db.Column(db.Float,nullable=False)
    address=db.Column(db.Float,nullable=False)
    income=db.Column(db.Float,nullable=False)
    education=db.Column(db.Float,nullable=False)
    employ=db.Column(db.Float,nullable=False)
    retire=db.Column(db.Float,nullable=False)
    gender=db.Column(db.Float,nullable=False)
    reside=db.Column(db.Float,nullable=False)
    class_knn=db.Column(db.Float,nullable=False)
        
    def _init__(region,tenure,age,marital,address,income,education,employ,retire,gender,reside,class_knn):
        self.id=id
        self.region=region
        self.tenure=tenure
        self.age=age
        self.marital=marital        
        self.address=address
        self.income=income
        self.education=education
        self.employ=employ
        self.retire=retire
        self.gender=gender
        self.reside=reside
        self.class_knn=class_knn

class Kmeans_table(db.Model):
    __tablename__='Kmeans'
    id=db.Column(db.Integer, primary_key=True)
    age=db.Column(db.Float,nullable=False)
    education=db.Column(db.Float,nullable=False)
    years_employ=db.Column(db.Float,nullable=False)
    income=db.Column(db.Float,nullable=False)
    card_debt=db.Column(db.Float,nullable=False)
    other_debt=db.Column(db.Float,nullable=False)
    defaulted=db.Column(db.Float,nullable=False)
    income_debt_ratio=db.Column(db.Float,nullable=False)
    class_kmean=db.Column(db.Float,nullable=False)
        
    def _init__(age,education,years_employ,income,card_debt,other_debt,defaulted,income_debt_ratio,class_kmean):
        self.id=id
        self.age=age
        self.education=education
        self.years_employ=year_employ
        self.income=income
        self.card_debt=card_debt
        self.other_debt=other_debt
        self.defaulted=defaulted
        self.income_debt_ratio=income_debt_ratio
        self.class_kmean=class_kmean

class DBSCAN_table(db.Model):
    __tablename__='DBSCAN'
    id=db.Column(db.Integer, primary_key=True)
    x_cood=db.Column(db.Float,nullable=False)
    y_cood=db.Column(db.Float,nullable=False)
    Temp_mean=db.Column(db.Float,nullable=False)
    Temp_max=db.Column(db.Float,nullable=False)
    Temp_min=db.Column(db.Float,nullable=False)
    class_DBSCAN=db.Column(db.Float,nullable=False)
        
    def _init__(x_cood,y_cood,Temp_mean,Temp_max,Temp_min,class_DBSCAN):
        self.id=id
        self.x_cood=x_cood
        self.y_cood=y_cood
        self.Temp_mean=Temp_mean
        self.Temp_min= Temp_min
        self.Temp_max=Temp_max
        self.class_DBSCAN=class_DBSCAN

class Naive_Bayes_table(db.Model):
    __tablename__='Naive_Bayes'
    id=db.Column(db.Integer, primary_key=True)
    tenure=db.Column(db.Float,nullable=False)
    age=db.Column(db.Float,nullable=False)
    address=db.Column(db.Float,nullable=False)
    income=db.Column(db.Float,nullable=False)
    education=db.Column(db.Float,nullable=False)
    employ=db.Column(db.Float,nullable=False)
    churn=db.Column(db.Float,nullable=False)
        
    def _init__(id,tenure,age,address,income,education,employ,churn):
        self.id=id
        self.tenure=tenure
        self.age=age
        self.address=address
        self.income=income
        self.education=education
        self.employ=employ
        self.churn=churn


@app.route('/', methods=['POST','GET'])

def index():
    if request.method=='POST':
        model=request.form['model']
        if model=='Linear':
            return redirect('/Linear')
        if model=='Non_Linear':
            return redirect('/Non_Linear')
        if model=='Logistic':
            return redirect('/Logistic')
        if model=='Decision_Tree':
            return redirect('/Decision_Tree')
        if model=='SVM_model':
            return redirect('/SVM_model')                
        if model=='Random_forest':
            return redirect('/Random_forest')
        if model=='PCA':
            return redirect('/PCA')
        if model=='KNN':
            return redirect('/KNN')
        if model=='Kmeans':
            return redirect('/Kmeans')
        if model=='DBSCAN':
            return redirect('/DBSCAN')
        if model=='Naive_Bayes':
            return redirect('/Naive_Bayes')                    
        else:
            return ('Index Error')
    else:
        return render_template('index.html')
    
@app.route('/Linear',methods=['POST','GET'])

def Model1():
    if request.method=='POST':
        engine=float(request.form['EngineSize'])
        try:
            co2 = Linear_Regression(engine)
            #print(co2)
        except:
            return 'Please enter Integer'

        entry=Linear_table(engine=engine,co2=co2)
        
        try:
            db.session.add(entry)
            db.session.commit()
            return redirect('/Linear')
        except:
            return 'Database error'   
    else:
        elements=Linear_table.query.order_by(Linear_table.id).all()
        return render_template('Linear.html',elements=elements)   


@app.route('/Linear/delete/<int:id>')

def delete1(id):
    entry_to_delete =Linear_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/Linear')
    except:
        return "Contact DBA"             


@app.route('/Linear/update/<int:id>',methods=['POST','GET'])

def get_update1(id):
    entry_to_update=Linear_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.engine=float(request.form['EngineSize'])
        entry_to_update.co2 = Linear_Regression(entry_to_update.engine)

        try:
            db.session.commit()
            return redirect('/Linear')
        except:
            return 'Contact DBA'

    else:
        return render_template('Linear_update.html',entry_to_update=entry_to_update)            

@app.route('/Non_Linear',methods=['POST','GET'])

def Model2():
    if request.method=='POST':
        year=float(request.form['year'])
        try:
            gdp=Non_Linear_Regression(year)
        except:
            return 'Please enter Integer'

        entry=Non_linear_table(year=year,gdp=gdp)
        try:
            db.session.add(entry)
            db.session.commit()
            return redirect('/Non_Linear')
        except:
            #print(year)
            return 'Database Error'   
    else:
        elements=Non_linear_table.query.order_by(Non_linear_table.id).all()
        return render_template('Non_Linear.html',elements=elements)

@app.route('/Non_Linear/delete/<int:id>')

def delete2(id):
    entry_to_delete =Non_linear_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/Non_Linear')
    except:
        return "Contact DBA"             


@app.route('/Non_Linear/update/<int:id>',methods=['POST','GET'])

def get_update2(id):
    entry_to_update=Non_linear_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.year=float(request.form['year'])
        entry_to_update.gdp = Linear_Regression(entry_to_update.year)

        try:
            db.session.commit()
            return redirect('/Non_Linear')
        except:
            return 'Contact DBA'

    else:
        return render_template('Non_Linear_update.html',entry_to_update=entry_to_update)           

@app.route('/Logistic',methods=['POST','GET'])

def Model3():
    if request.method=='POST':
        tenure=float(request.form['tenure'])
        age=float(request.form['age'])
        address=float(request.form['address'])
        income=float(request.form['income'])
        education=float(request.form['education'])
        employ=float(request.form['employ'])
        equip=float(request.form['equip'])
        payload=[tenure,age,address,income,education,employ,equip]
        payload=np.array(payload)

        try:
            churn = Logistic_Regression(payload)
        except:
            return 'Please enter Float' 

        entry=Logistic_table(tenure=tenure,age=age,address=address,income=income,education=education,employ=employ,equip=equip,churn=churn)

        try:
            db.session.add(entry)
            db.session.commit()
            return redirect('/Logistic')
        except:
            #print(year)
            return 'Database Error'      
    else:
        elements=Logistic_table.query.order_by(Logistic_table.id).all()
        return render_template('Logistic.html',elements=elements)

@app.route('/Logistic/delete/<int:id>')

def delete3(id):
    entry_to_delete =Logistic_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/Logistic')
    except:
        return "Contact DBA"             


@app.route('/Logistic/update/<int:id>',methods=['POST','GET'])

def get_update3(id):
    entry_to_update=Logistic_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.tenure=float(request.form['tenure'])
        entry_to_update.age=float(request.form['age'])
        entry_to_update.address=float(request.form['address'])
        entry_to_update.income=float(request.form['income'])
        entry_to_update.education=float(request.form['education'])
        entry_to_update.employ=float(request.form['employ'])
        entry_to_update.equip=float(request.form['equip'])
        payload=[entry_to_update.tenure,entry_to_update.age,entry_to_update.address,entry_to_update.income,entry_to_update.education,entry_to_update.employ,entry_to_update.equip]
        payload=np.array(payload)
        entry_to_update.churn=Logistic_Regression(payload)

        try:
            db.session.commit()
            return redirect('/Logistic')
        except:
            return 'Contact DBA'

    else:
        return render_template('Logistic_update.html',entry_to_update=entry_to_update)           


@app.route('/Decision_Tree',methods=['POST','GET'])

def Model4():
    if request.method=='POST':
        age=float(request.form['age'])
        sex=float(request.form['sex'])
        bp=float(request.form['bp'])
        cholestrol=float(request.form['cholestrol'])
        na_to_k=float(request.form['na_to_k'])
        payload=[age,sex,bp,cholestrol,na_to_k]
        payload=np.array(payload)

        try:
            drug = Decision_Tree(payload)
        except:
            return 'Please enter Float' 

        entry=Decision_table(age=age,sex=sex,bp=bp,cholestrol=cholestrol,na_to_k=na_to_k,drug=drug)

        try:
            db.session.add(entry)
            db.session.commit()
            return redirect('/Decision_Tree')
        except:
            #print(year)
            return 'Database Error'      
    else:
        elements=Decision_table.query.order_by(Decision_table.id).all()
        return render_template('Decision_Tree.html',elements=elements)

@app.route('/Decision_Tree/delete/<int:id>')

def delete4(id):
    entry_to_delete =Decision_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/Decision_Tree')
    except:
        return "Contact DBA"             


@app.route('/Decision_Tree/update/<int:id>',methods=['POST','GET'])

def get_update4(id):
    entry_to_update=Decision_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.age=float(request.form['age'])
        entry_to_update.sex=float(request.form['sex'])
        entry_to_update.bp=float(request.form['bp'])
        entry_to_update.cholestrol=float(request.form['cholestrol'])
        entry_to_update.na_to_k=float(request.form['na_to_k'])
        payload=[age,sex,bp,cholestrol,na_to_k]
        payload=np.array(payload)
        entry_to_update.drug=Decision_Tree(payload)

        try:
            db.session.commit()
            return redirect('/Decision_Tree')
        except:
            return 'Contact DBA'

    else:
        return render_template('Decision_Tree_update.html',entry_to_update=entry_to_update)           

@app.route('/SVM_model',methods=['POST','GET'])

def Model5():
    if request.method=='POST':
        clump=float(request.form['clump'])
        unifsize=float(request.form['unifsize'])
        unifshape=float(request.form['unifshape'])
        margadh=float(request.form['margadh'])
        singepisize=float(request.form['singepisize'])
        barenuc=float(request.form['barenuc'])
        blandchrom=float(request.form['blandchrom'])
        normnucl=float(request.form['normnucl'])
        mit=float(request.form['mit'])
        payload=[clump,unifsize,unifshape,margadh,singepisize,barenuc,blandchrom,normnucl,mit]
        payload=np.array(payload)
        #print(payload)

        try:
            cell_class = SVM_model(payload)
            #print(cell_class)
        except:
            return 'Please enter Float' 

        entry=SVM_table(clump=clump,unifsize=unifsize,unifshape=unifshape,margadh=margadh,singepisize=singepisize,barenuc=barenuc,blandchrom=blandchrom,normnucl=normnucl,mit=mit,cell_class=cell_class)

        try:
            db.session.add(entry)
            print('Clear')
            db.session.commit()
            print('Clear')
            return redirect('/SVM_model')
        except:
            #print(year)
            return 'Database Error5'      
    else:
        elements=SVM_table.query.order_by(SVM_table.id).all()
        return render_template('SVM_model.html',elements=elements)

@app.route('/SVM_model/delete/<int:id>')

def delete5(id):
    entry_to_delete =SVM_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/SVM_model')
    except:
        return "Contact DBA"             


@app.route('/SVM_model/update/<int:id>',methods=['POST','GET'])

def get_update5(id):
    entry_to_update=SVM_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.clump=float(request.form['clump'])
        entry_to_update.unifsize=float(request.form['unifsize'])
        entry_to_update.unifshape=float(request.form['unifshape'])
        entry_to_update.margadh=float(request.form['margadh'])
        entry_to_update.singepisize=float(request.form['singepisize'])
        entry_to_update.barenuc=float(request.form['barenuc'])
        entry_to_update.blandchrom=float(request.form['blandchrom'])
        entry_to_update.normnucl=float(request.form['normnucl'])
        entry_to_update.mit=float(request.form['mit'])
        payload=[entry_to_update.clump,entry_to_update.unifsize,entry_to_update.unifshape,entry_to_update.margadh,entry_to_update.singepisize,entry_to_update.barenuc,entry_to_update.blandchrom,entry_to_update.normnucl,entry_to_update.mit]
        payload=np.array(payload)
        entry_to_update.cell_class=SVM_model(payload)

        try:
            db.session.commit()
            return redirect('/SVM_model')
        except:
            return 'Contact DBA'

    else:
        return render_template('SVM_model_update.html',entry_to_update=entry_to_update)           

@app.route('/Random_forest',methods=['POST','GET'])

def Model6():
    if request.method=='POST':
        clump=float(request.form['clump'])
        unifsize=float(request.form['unifsize'])
        unifshape=float(request.form['unifshape'])
        margadh=float(request.form['margadh'])
        singepisize=float(request.form['singepisize'])
        barenuc=float(request.form['barenuc'])
        blandchrom=float(request.form['blandchrom'])
        normnucl=float(request.form['normnucl'])
        mit=float(request.form['mit'])
        payload=[clump,unifsize,unifshape,margadh,singepisize,barenuc,blandchrom,normnucl,mit]
        payload=np.array(payload)
        #print(payload)

        try:
            cell_class = Random_forest(payload)
            #print(cell_class)
        except:
            return 'Please enter Float' 

        entry=Random_forest_table(clump=clump,unifsize=unifsize,unifshape=unifshape,margadh=margadh,singepisize=singepisize,barenuc=barenuc,blandchrom=blandchrom,normnucl=normnucl,mit=mit,cell_class=cell_class)

        try:
            db.session.add(entry)
            print('Clear')
            db.session.commit()
            print('Clear')
            return redirect('/Random_forest')
        except:
            #print(year)
            return 'Database Error5'      
    else:
        elements=Random_forest_table.query.order_by(Random_forest_table.id).all()
        return render_template('Random_forest.html',elements=elements)

@app.route('/Random_forest/delete/<int:id>')

def delete6(id):
    entry_to_delete =Random_forest_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/Random_forest')
    except:
        return "Contact DBA"             


@app.route('/Random_forest/update/<int:id>',methods=['POST','GET'])

def get_update6(id):
    entry_to_update=Random_forest_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.clump=float(request.form['clump'])
        entry_to_update.unifsize=float(request.form['unifsize'])
        entry_to_update.unifshape=float(request.form['unifshape'])
        entry_to_update.margadh=float(request.form['margadh'])
        entry_to_update.singepisize=float(request.form['singepisize'])
        entry_to_update.barenuc=float(request.form['barenuc'])
        entry_to_update.blandchrom=float(request.form['blandchrom'])
        entry_to_update.normnucl=float(request.form['normnucl'])
        entry_to_update.mit=float(request.form['mit'])
        payload=[entry_to_update.clump,entry_to_update.unifsize,entry_to_update.unifshape,entry_to_update.margadh,entry_to_update.singepisize,entry_to_update.barenuc,entry_to_update.blandchrom,entry_to_update.normnucl,entry_to_update.mit]
        payload=np.array(payload)
        entry_to_update.cell_class=Random_forest(payload)

        try:
            db.session.commit()
            return redirect('/Random_forest')
        except:
            return 'Contact DBA'

    else:
        return render_template('Random_forest_update.html',entry_to_update=entry_to_update)           

@app.route('/PCA',methods=['POST','GET'])

def Model7():
    if request.method=='POST':
        clump=float(request.form['clump'])
        unifsize=float(request.form['unifsize'])
        unifshape=float(request.form['unifshape'])
        margadh=float(request.form['margadh'])
        singepisize=float(request.form['singepisize'])
        barenuc=float(request.form['barenuc'])
        blandchrom=float(request.form['blandchrom'])
        normnucl=float(request.form['normnucl'])
        mit=float(request.form['mit'])
        payload=[clump,unifsize,unifshape,margadh,singepisize,barenuc,blandchrom,normnucl,mit]
        payload=np.array(payload)
        #print(payload)

        try:
            cell_class = PCA(payload)
            #print(cell_class)
        except:
            return 'Please enter Float' 

        entry=PCA_table(clump=clump,unifsize=unifsize,unifshape=unifshape,margadh=margadh,singepisize=singepisize,barenuc=barenuc,blandchrom=blandchrom,normnucl=normnucl,mit=mit,cell_class=cell_class)

        try:
            db.session.add(entry)
            print('Clear')
            db.session.commit()
            print('Clear')
            return redirect('/PCA')
        except:
            #print(year)
            return 'Database Error5'      
    else:
        elements=PCA_table.query.order_by(PCA_table.id).all()
        return render_template('PCA.html',elements=elements)

@app.route('/PCA/delete/<int:id>')

def delete7(id):
    entry_to_delete =PCA_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/PCA')
    except:
        return "Contact DBA"             


@app.route('/PCA/update/<int:id>',methods=['POST','GET'])

def get_update7(id):
    entry_to_update=PCA_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.clump=float(request.form['clump'])
        entry_to_update.unifsize=float(request.form['unifsize'])
        entry_to_update.unifshape=float(request.form['unifshape'])
        entry_to_update.margadh=float(request.form['margadh'])
        entry_to_update.singepisize=float(request.form['singepisize'])
        entry_to_update.barenuc=float(request.form['barenuc'])
        entry_to_update.blandchrom=float(request.form['blandchrom'])
        entry_to_update.normnucl=float(request.form['normnucl'])
        entry_to_update.mit=float(request.form['mit'])
        payload=[entry_to_update.clump,entry_to_update.unifsize,entry_to_update.unifshape,entry_to_update.margadh,entry_to_update.singepisize,entry_to_update.barenuc,entry_to_update.blandchrom,entry_to_update.normnucl,entry_to_update.mit]
        payload=np.array(payload)
        entry_to_update.cell_class=PCA(payload)

        try:
            db.session.commit()
            return redirect('/PCA')
        except:
            return 'Contact DBA'

    else:
        return render_template('PCA_update.html',entry_to_update=entry_to_update)           

@app.route('/KNN',methods=['POST','GET'])

def Model8():
    if request.method=='POST':
        region=float(request.form['region'])
        tenure=float(request.form['tenure'])
        age=float(request.form['age'])
        marital=float(request.form['marital'])
        address=float(request.form['address'])
        income=float(request.form['income'])
        education=float(request.form['education'])
        employ=float(request.form['employ'])
        retire=float(request.form['retire'])
        gender=float(request.form['gender'])
        reside=float(request.form['reside'])
        payload=[region,tenure,age,marital,address,income,education,employ,retire,gender,reside]
        payload=np.array(payload)
        #print(payload)

        try:
            class_knn = KNN(payload)
            
        except:
            return 'Please enter Float' 

        entry=KNN_table(region=region,tenure=tenure,age=age,marital=marital,address=address,income=income,education=education,employ=employ,retire=retire,gender=gender,reside=reside,class_knn=class_knn)

        try:
            db.session.add(entry)
            #print('Clear')
            db.session.commit()
            #print('Clear')
            return redirect('/KNN')
        except:
            return 'Database Error8'      
    else:
        elements=KNN_table.query.order_by(KNN_table.id).all()
        return render_template('KNN.html',elements=elements)

@app.route('/KNN/delete/<int:id>')

def delete8(id):
    entry_to_delete =KNN_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/KNN')
    except:
        return "Contact DBA"             


@app.route('/KNN/update/<int:id>',methods=['POST','GET'])

def get_update8(id):
    entry_to_update=KNN_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.region=float(request.form['region'])
        entry_to_update.tenure=float(request.form['tenure'])
        entry_to_update.age=float(request.form['age'])
        entry_to_update.marital=float(request.form['marital'])
        entry_to_update.address=float(request.form['address'])
        entry_to_update.income=float(request.form['income'])
        entry_to_update.education=float(request.form['education'])
        entry_to_update.employ=float(request.form['employ'])
        entry_to_update.retire=float(request.form['retire'])
        entry_to_update.gender=float(request.form['gendern'])
        entry_to_update.reside=float(request.form['reside'])
        payload=[entry_to_update.region,entry_to_update.tenure,entry_to_update.age,entry_to_update.marital,entry_to_update.address,entry_to_update.income,entry_to_update.education,entry_to_update.employ,entry_to_update.retire,entry_to_update.gender,entry_to_update.reside]
        payload=np.array(payload)
        entry_to_update.class_knn=KNN(payload)

        try:
            db.session.commit()
            return redirect('/KNN')
        except:
            return 'Contact DBA'

    else:
        return render_template('KNN_update.html',entry_to_update=entry_to_update)           

@app.route('/Kmeans',methods=['POST','GET'])

def Model9():
    if request.method=='POST':
        age=float(request.form['age'])
        education=float(request.form['education'])
        years_employ=float(request.form['years_employ'])
        income=float(request.form['income'])
        card_debt=float(request.form['card_debt'])
        other_debt=float(request.form['other_debt'])
        defaulted=float(request.form['defaulted'])
        income_debt_ratio=float(request.form['income_debt_ratio'])
        payload=[age,education,years_employ,income,card_debt,other_debt,defaulted,income_debt_ratio]  
        payload=np.array(payload)
        #print(payload)

        try:
            class_kmean = Kmeans(payload)
            print(class_kmean)
        except:
            return 'Please enter Float' 

        entry=Kmeans_table(age=age,education=education,years_employ=years_employ,income=income,card_debt=card_debt,other_debt=other_debt,defaulted=defaulted,income_debt_ratio=income_debt_ratio,class_kmean=class_kmean)

        try:
            db.session.add(entry)
            #print('Clear')
            db.session.commit()
            #print('Clear')
            return redirect('/Kmeans')
        except:
            #print(year)
            return 'Database Error9'      
    else:
        elements=Kmeans_table.query.order_by(Kmeans_table.id).all()
        return render_template('Kmeans.html',elements=elements)

@app.route('/Kmeans/delete/<int:id>')

def delete9(id):
    entry_to_delete =Kmeans_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/Kmeans')
    except:
        return "Contact DBA"             


@app.route('/Kmeans/update/<int:id>',methods=['POST','GET'])

def get_update9(id):
    entry_to_update=Kmeans_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.age=float(request.form['age'])
        entry_to_update.education=float(request.form['education'])
        entry_to_update.years_employ=float(request.form['years_employ'])
        entry_to_update.income=dfloat(request.form['income'])
        entry_to_update.card_debt=dfloat(request.form['card_debt'])
        entry_to_update.other_debt=float(request.form['other_debt'])
        entry_to_update.defaulted=dfloat(request.form['defaulted'])
        entry_to_update.income_debt_ratio=float(request.form['income_debt_ratio'])
        payload=[entry_to_update.age,entry_to_update.education,entry_to_update.years_employ,entry_to_update.income,entry_to_update.card_debt,entry_to_update.other_debt,entry_to_update.defaulted,entry_to_update.income_debt_ratio]  
        payload=np.array(payload)
        entry_to_update.class_kmean=Kmeans(payload)

        try:
            db.session.commit()
            return redirect('/Kmeans')
        except:
            return 'Contact DBA'

    else:
        return render_template('Kmeans_update.html',entry_to_update=entry_to_update)           

@app.route('/DBSCAN',methods=['POST','GET'])

def Model10():
    if request.method=='POST':
        x_cood=float(request.form['x_cood'])
        y_cood=float(request.form['y_cood'])
        Temp_mean=float(request.form['Temp_mean'])
        Temp_max=float(request.form['Temp_max'])
        Temp_min=float(request.form['Temp_min'])
        payload=[x_cood,y_cood,Temp_mean,Temp_max,Temp_min]
        payload=np.array(payload)
        
        try:
            class_DBSCAN = DBSCAN(payload)            
        except:
            return 'Please enter Float' 

        entry=DBSCAN_table(x_cood=x_cood,y_cood=y_cood,Temp_mean=Temp_mean,Temp_max=Temp_max,Temp_min=Temp_min,class_DBSCAN=class_DBSCAN)

        try:
            db.session.add(entry)
            print('Clear')
            db.session.commit()
            print('Clear')
            return redirect('/DBSCAN')
        except:
            #print(year)
            return 'Database Error10'      
    else:
        elements=DBSCAN_table.query.order_by(DBSCAN_table.id).all()
        return render_template('DBSCAN.html',elements=elements)

@app.route('/DBSCAN/delete/<int:id>')

def delete10(id):
    entry_to_delete =DBSCAN_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/DBSCAN')
    except:
        return "Contact DBA"             


@app.route('/DBSCAN/update/<int:id>',methods=['POST','GET'])

def get_update10(id):
    entry_to_update=DBSCAN_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.x_cood=float(request.form['x_cood'])
        entry_to_update.y_cood=float(request.form['y_cood'])
        entry_to_update.Temp_mean=float(request.form['Temp_mean'])
        entry_to_update.Temp_max=float(request.form['Temp_max'])
        entry_to_update.Temp_min=float(request.form['Temp_min'])
        payload=[entry_to_update.x_cood,entry_to_update.y_cood,entry_to_update.Temp_mean,entry_to_update.Temp_max,entry_to_update.Temp_min]
        payload=np.array(payload)
        entry_to_update.class_DBSCAN=DBSCAN(payload)

        try:
            db.session.commit()
            return redirect('/DBSCAN')
        except:
            return 'Contact DBA'

    else:
        return render_template('DBSCAN_update.html',entry_to_update=entry_to_update)           

@app.route('/Naive_Bayes',methods=['POST','GET'])

def Model11():
    if request.method=='POST':
        tenure=float(request.form['tenure'])
        age=float(request.form['age'])
        address=float(request.form['address'])
        income=float(request.form['income'])
        education=float(request.form['education'])
        employ=float(request.form['employ'])
        payload=[tenure,age,address,income,education,employ]
        payload=np.array(payload)

        try:
            churn = Naive_Bayes(payload)
        except:
            return 'Please enter Float' 

        entry=Naive_Bayes_table(tenure=tenure,age=age,address=address,income=income,education=education,employ=employ,churn=churn)

        try:
            db.session.add(entry)
            db.session.commit()
            return redirect('/Naive_Bayes')
        except:
            #print(year)
            return 'Database Error11'      
    else:
        elements=Naive_Bayes_table.query.order_by(Naive_Bayes_table.id).all()
        return render_template('Naive_Bayes.html',elements=elements)

@app.route('/Naive_Bayes/delete/<int:id>')

def delete11(id):
    entry_to_delete =Naive_Bayes_table.query.get_or_404(id)

    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/Naive_Bayes')
    except:
        return "Contact DBA"             


@app.route('/Naive_Bayes/update/<int:id>',methods=['POST','GET'])

def get_update11(id):
    entry_to_update=Naive_Bayes_table.query.get_or_404(id)

    if request.method=='POST':
        entry_to_update.tenure=float(request.form['tenure'])
        entry_to_update.age=float(request.form['age'])
        entry_to_update.address=float(request.form['address'])
        entry_to_update.income=float(request.form['income'])
        entry_to_update.education=float(request.form['education'])
        entry_to_update.employ=float(request.form['employ'])
        payload=[entry_to_update.tenure,entry_to_update.age,entry_to_update.address,entry_to_update.income,entry_to_update.education,entry_to_update.employ]
        payload=np.array(payload)
        entry_to_update.churn=Naive_Bayes(payload)

        try:
            db.session.commit()
            return redirect('/Naive_Bayes')
        except:
            return 'Contact DBA'

    else:
        return render_template('Naive_Bayes_update.html',entry_to_update=entry_to_update)           


if __name__  == '__main__':
    app.run(debug=True,host='0.0.0.0')
