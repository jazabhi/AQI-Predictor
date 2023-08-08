from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('forest_model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("air_quality.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output <= str(50):
        return render_template('air_quality.html', pred='Good\n Air quality index of the city is {}'.format(output))
    elif output > str(50) and output <= str(100):
        return render_template('air_quality.html', pred='Moderate\n Air quality index of the city is {}'.format(output))
    elif output > str(100) and output <= str(200):
        return render_template('air_quality.html', pred='Poor\n Air quality index of the city is {}'.format(output))
    elif output > str(200) and output <= str(300):
        return render_template('air_quality.html',
                               pred='Unhealthy\n Air quality index of the city is {}'.format(output))
    elif output > str(300) and output <= str(400):
        return render_template('air_quality.html',
                               pred='Very unhealthy\n Air quality index of the city is {}'.format(output))
    elif output > str(400):
        return render_template('air_quality.html', pred='Hazardous\n Air quality index of the city is {}'.format(output))


@app.route('/state', methods=['POST', 'GET'])
def state():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        print(int_features)
        print(final)
        prediction = model.predict(final)
        output = prediction[0]
        print(type(output))

        if output <= 50:
            return render_template('state2.html', pred='Good, Air quality index of the city is {}'.format(output))
        elif output > 50 and output <= 120:
            return render_template('state2.html', pred='Moderate, Air quality index of the city is {}'.format(output))
        elif output > 120 and output <= 200:
            return render_template('state2.html', pred='Poor, Air quality index of the city is {}'.format(output))
        elif output > 200 and output <= 300:
            return render_template('state2.html', pred='Unhealthy, Air quality index of the city is {}'.format(output))
        elif output > 300 and output <= 400:
            return render_template('state2.html',
                                   pred='Very unhealthy, Air quality index of the city is {}'.format(output))
        elif output > 400:
            return render_template('state2.html', pred='Hazardous,Air quality index of the city is {}'.format(output))
    print("Check")
    return render_template('state2.html')

if __name__ == '__main__':
    app.run(debug=True)
