let classifier;

let imghold, img, input, button;

let label, confidence

function preload() {
    classifier = ml5.imageClassifier('MobileNet');
}

function getimg() {
    if (imghold != input.value()) {
        img = loadImage(input.value(), render);
    }
    imghold = input.value();
}

function render() {
    var canvas = createCanvas(img.width, img.height);
    canvas.style('margin', '3em');
    if (displayWidth < 768) {
        canvas.style('width', '90%');
        canvas.style('height', '90%');
    };
    image(img, 0, 0, width, height);
    classifier.classify(img, gotResult);
}

function setup() {
    var instructor = createDiv('Paste a link of picture here.');
    instructor.style('margin-left', '20px');
    instructor.style('margin-bottom', '10px');
    input = createInput().style('margin-left', '20px');
    button = createButton('submit').style('margin-left', '20px');
    button.mouseClicked(getimg);
    label = createDiv('');
    confidence = createDiv('');
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
    } else {
        label.html('Label : ' + results[0].label);
        confidence.html('Confidence : ' + (results[0].confidence * 100).toFixed(2) + '%');

        label.style('font-size', '30px');
        label.style('margin-top', '20px');
        label.style('margin-left', '20px');
        confidence.style('font-size', '20px');
        confidence.style('margin-left', '20px');
    }
}

