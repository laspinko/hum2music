var wavesurfer = WaveSurfer.create({
    container: document.querySelector('#waveform'),
	waveColor: 'rgb(150, 150, 94)',
    progressColor: 'rgb(255, 234, 117)',
    cursorColor: 'rgba(94, 60, 16, 0.5)',
    barWidth: 3,
    barRadius: 3,
    cursorWidth: 3,
    height: 60,
    barGap: 3,
    backend: 'MediaElement',
    plugins: [
        WaveSurfer.regions.create({
            regionsMinLength: 2,
            regions: [

            ],
            dragSelection: {
				slop: 10000
            }
        })
    ]
});


const c = document.getElementById("circleBtn");
const u = document.getElementById("uploadBtn");
const t = document.getElementById("trimBar");
const b = document.getElementById("buttonContainer");
const n = document.getElementById("nextBtn");
const tt = document.getElementById("trimTable");
const s = document.getElementById("shareButtonContainer");
const l = document.getElementById("logo");

var up = false;
var finalPage = false;
var trimCount = 0;

function moveUp() {
	if(up == false){
		c.style.transform = `translateY(-25vh) scale(0.7)`;
		u.classList.add("hidden"); 
		u.classList.remove("visible"); 

		t.classList.remove("hidden"); 
		t.classList.add("visible"); 

		b.classList.remove("hidden"); 
		b.classList.add("visible"); 

		n.classList.remove("hidden"); 
		n.classList.add("visible"); 

		tt.classList.remove("hidden"); 
		tt.classList.add("visible"); 
		
		l.classList.add("hidden"); 
		l.classList.remove("visible"); 

		up = true;
	}
	else {
		c.style.transform = `translate(0, 0)`;
		u.classList.add("visible"); 
		u.classList.remove("hidden"); 

		t.classList.add("hidden"); 
		t.classList.remove("visible"); 

		b.classList.add("hidden"); 
		b.classList.remove("visible"); 

		n.classList.add("hidden"); 
		n.classList.remove("visible"); 

		tt.classList.add("hidden"); 
		tt.classList.remove("visible"); 
		
		l.classList.remove("hidden"); 
		l.classList.add("visible"); 

		up = false
	}
}
function playPause(){
	wavesurfer.playPause()
}
function discard(){
	document.getElementById("trimTable").innerHTML = "";
}
function trim(){
	var newTrim = document.createElement("li");
	if(trimCount==0){
		var trimmedAudio = document.createTextNode("00:10:03 - 00:18:32　　");
		newTrim.appendChild(trimmedAudio);
		trimCount += 1;
	}
	else{
		var trimmedAudio = document.createTextNode("00:22:81 - 00:31:46　　");
		newTrim.appendChild(trimmedAudio);
	}


	var checkbox = document.createElement('input');
	checkbox.type = "checkbox";
	checkbox.value = 1;
	checkbox.name = "todo[]";
	newTrim.appendChild(checkbox);


	newTrim.className = "tableRow";
	document.getElementById("trimTable").appendChild(newTrim)
}

function nextPage() {
	if(finalPage == false){
		n.style.transform = 'translateY(-70vh) scale(1.1)';

		c.style.transform = `translateY(-75vh) scale(0.4)`;

		t.classList.add("hidden"); 
		t.classList.remove("visible"); 

		tt.classList.add("hidden"); 
		tt.classList.remove("visible"); 

		b.classList.add("hidden"); 
		b.classList.remove("visible"); 

		s.classList.add("visible"); 
		s.classList.remove("hidden"); 
		s.style.transform = `translateY(-35vh)`;

		finalPage = true;

		n.innerHTML = "◂ Previous Page"
	}
	else {
		
		n.style.transform = 'translate(0,0)';

		c.style.transform = `translateY(-25vh) scale(0.7)`;
		t.classList.remove("hidden"); 
		t.classList.add("visible"); 

		tt.classList.remove("hidden"); 
		tt.classList.add("visible"); 

		b.classList.remove("hidden"); 
		b.classList.add("visible"); 

		s.classList.remove("visible"); 
		s.classList.add("hidden"); 
		s.style.transform = `translate(0,0)`;


		finalPage = false
		n.innerHTML = "Final Step"
	}
}

wavesurfer.load('test_audio.mp3');








