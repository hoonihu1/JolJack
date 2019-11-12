document.getElementById("camera").onclick = function() {
    document.getElementById("camera").style.display = "none";
    displayOut();
    displayOn();
};

function displayOut(){
    document.getElementById("player").style.display = "none";
}

function displayOn(){
    document.getElementById("gotoserver").style.display = "block";
    document.getElementById("snapshot").style.display = "block";
}