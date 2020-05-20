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
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pickle


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
        
    def _init__(id,age,sex,bp,cholestrol,na_to_k,drug):
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
        
    def _init__(clump,unifsize,unifshape,margadh,singepisize,barenuc,blandchrom,normnucl,mit,cell_class):
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
        
    def _init__(clump,unifsize,unifshape,margadh,singepisize,barenuc,blandchrom,normnucl,mit,cell_class):
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


if __name__  == '__main__':
    app.run(debug=True,host='0.0.0.0')
