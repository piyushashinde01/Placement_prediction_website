from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('lr.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', pred='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Update the names of the input fields to match the HTML form
        int_features = [float(request.form[f'input{i}']) for i in range(2, 12)]
        final = [np.array(int_features)]
        prediction = model.predict(final)

        # Assuming prediction is an array with probabilities for each class
        output = prediction[0]

        if output > 0.5:
            pred_message = f'Probability is {output:.2f}. Placement Likely'
        else:
            pred_message = f'Probability is {output:.2f}. Placement Unlikely'

    except Exception as e:
        print(e)
        print(f"An error occurred: {str(e)}")
        pred_message = 'An error occurred.'

    return render_template('index.html', pred=pred_message)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feed')
def feed():
    return render_template('feed.html')

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request
# from flask_sqlalchemy import SQLAlchemy
# import pickle
# import numpy as np

# app = Flask(__name__

# # Define a model for the database
# class Prediction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     probability = db.Column(db.Float)

# # Load the machine learning model
# model = pickle.load(open('lr.pkl', 'rb'))

# @app.route('/')
# def index():
#     return render_template('index.html', pred='')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         int_features = [float(request.form[f'input{i}']) for i in range(2, 12)]
#         final = [np.array(int_features)]
#         prediction = model.predict(final)

#         output = prediction[0]

#         # Save the prediction to the database
#         new_prediction = Prediction(probability=output)
#         db.session.add(new_prediction)
#         db.session.commit()

#         if output > 0.5:
#             pred_message = f'Probability is {output:.2f}. Placement Likely'
#         else:
#             pred_message = f'Probability is {output:.2f}. Placement Unlikely'

#     except Exception as e:
#         print(e)
#         print(f"An error occurred: {str(e)}")
#         pred_message = 'An error occurred.'

#     return render_template('index.html', pred=pred_message)

# @app.route('/home')
# def home():
#     return render_template('home.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/feed')
# def feed():
#     # Retrieve predictions from the database
#     predictions = Prediction.query.all()
#     return render_template('feed.html', predictions=predictions)

# if __name__ == '__main__':
#     # Create the database tables before running the app
#     db.create_all()
#     app.run(debug=True)



# from flask import Flask, render_template

# app = Flask(__name__)

# # Load your model
# # model = pickle.load(open('lr.pkl', 'rb'))

# @app.route('/')
# def index():
#     return render_template('index.html', pred='')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Your existing code for prediction
#         # ...

#     except Exception as e:
#         print(e)
#         print(f"An error occurred: {str(e)}")
#         pred_message = 'An error occurred.'

#     return render_template('index.html', pred=pred_message)

# @app.route('/home')
# def home():
#     return render_template('home.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# if __name__ == '__main__':
#     app.run(debug=True)
