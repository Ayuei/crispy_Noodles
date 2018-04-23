function more(myPage, windowName, myWidth, myHeight, resize){
	LeftPosition= (screen.width*0.5)-(myWidth*0.5);
	TopPosition= (screen.height*0.4)-(myHeight*0.5);
	window.open(myPage, windowName, 'width='+myWidth+',height='+myHeight+',top='+TopPosition+',left='+LeftPosition+',scrollbars=yes,location=no,directories=no,status=no,menubar=no,toolbar=no,resizable=' + resize);
}

function changeNav(navName){
	if (navName != 'nav_a') document.getElementById('nav_a').className='hiddenTable';
	if (navName != 'nav_b') document.getElementById('nav_b').className='hiddenTable';
	document.getElementById(navName).className='visibleTable';
}

function expand(expandId,totalElem){
	// ************** di/expand only one elemend
	if (expandId != 'open' && expandId != 'close') {
		if (document.getElementById('arr' + expandId).src.indexOf('images/arrw_open.gif') > -1){
			document.getElementById('arr' + expandId).src = 'images/arrw_close.gif';
			document.getElementById('dynDiv' + expandId).className = 'hiddenTable';
		} else {
			document.getElementById('arr' + expandId).src = 'images/arrw_open.gif';
			document.getElementById('dynDiv' + expandId).className = 'visibleTable';
		}
	} else { // ********** di/expand all elements
		if (expandId == 'open') tmpVar = 'visible';
		else tmpVar = 'hidden';
		for (i = 1; i < totalElem + 1 ; i++) {
			document.getElementById('dynDiv' + i).className = tmpVar +'Table';
			document.getElementById('arr' + i).src = 'images/arrw_' + expandId + '.gif';
		}
	}
}

function disableForm(theform) {
	for(var i=0; i<theform.elements.length; i++){
     	if(theform.elements[i].value == "Complete"){
			theform.elements[i].disabled = true;
			theform.ItsDisabled.value="Yes";
        }
    }   
}