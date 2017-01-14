(function(){
	
	'use strict';
	
	angular
		.module('fitbitMLApp')
		.controller('profileController', profileController);
		
		function profileController($http){
			var vm = this;
			
			vm.message = "Yo there"
			
		}
	
})