(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory(require("d3"));
	else if(typeof define === 'function' && define.amd)
		define(["d3"], factory);
	else if(typeof exports === 'object')
		exports["slid3r"] = factory(require("d3"));
	else
		root["slid3r"] = factory(root["d3"]);
})(this, function(__WEBPACK_EXTERNAL_MODULE_4__) {
return /******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 0);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _typeof = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? function (obj) { return typeof obj; } : function (obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; };

var _selectAppend = __webpack_require__(1);

var _selectAppend2 = _interopRequireDefault(_selectAppend);

var _findClosestTickColor = __webpack_require__(2);

var _findClosestTickColor2 = _interopRequireDefault(_findClosestTickColor);

var _styles = __webpack_require__(3);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var d3 = __webpack_require__(4);


// Calculates the beta function between alpha and beta
/**
 * @return {object} - A slider object
 */
function slid3r() {
  // Defaults
  var sliderRange = [0, 10],
      sliderWidth = 250,
      onDone = function onDone(x) {
    return console.log("done dragging", x);
  },
      onDrag = function onDrag(x) {
    return null;
  },
      label = "choose value",
      startPos = 0,
      xPos = 0.5,
      yPos = 0.5,
      intClamp = true,
      // clamp handle to nearest int?
  numTicks = 10,
      customTicks = null,
      font = "optima",
      handleColor = "white",
      vertical = false,
      transitionLength = 10;

  // Calculates the beta function between alpha and beta
  /**
     * @param {object} sel - A selection from d3
     * @return {object} - A slider
     */
  function drawSlider(sel) {
    var trans = d3.transition("sliderTrans").duration(transitionLength);

    var xScale = d3.scaleLinear().domain(sliderRange).range([0, sliderWidth]).clamp(true);

    var getValue = function getValue(eventX) {
      return xScale.invert(eventX);
    };

    // tick logic.
    var tickFormat = xScale.tickFormat(5, intClamp ? ",.0f" : "f");
    var tickPositions = xScale.ticks(numTicks).map(tickFormat);

    // use user custom ticks info if provided, otherwise generate with d3.
    // check if custom ticks are just an array or are the more complex object version
    var customTickSimple = customTicks && _typeof(customTicks[0]) !== "object";
    var customColors = customTicks && customTicks[0].color;

    var tickData = !customTicks ? tickPositions.map(function (label) {
      return { label: label, pos: label, color: handleColor };
    }) : customTickSimple ? customTicks.map(function (d) {
      return { label: d, pos: d, color: handleColor };
    }) : customTicks.map(function (d) {
      return Object.assign({ color: handleColor }, d);
    });

    var slider = sel.attr("transform", "translate(" + xPos + ", " + yPos + ")");

    if (vertical) {
      slider.attr("transform", "rotate(90)");
    }

    var track = (0, _selectAppend2.default)(slider, "line", ".track").attr("x1", xScale.range()[0]).attr("x2", xScale.range()[1]);

    var trackInset = (0, _selectAppend2.default)(slider, "line", ".track-inset").attr("x1", xScale.range()[0]).attr("x2", xScale.range()[1]);

    var trackOverlay = (0, _selectAppend2.default)(slider, "line", ".track-overlay").attr("x1", xScale.range()[0]).attr("x2", xScale.range()[1]);

    trackOverlay.call(d3.drag().on("start.interrupt", function () {
      slider.interrupt();
    }).on("start drag", dragBehavior).on("end", finishBehavior));

    var handle = (0, _selectAppend2.default)(slider, "circle", ".handle").style("pointer-events", "none").attr("class", "handle").attr("r", 8).attr("fill", customColors ? (0, _findClosestTickColor2.default)(tickData, startPos) : handleColor).attr("cx", xScale(startPos));

    (0, _selectAppend2.default)(slider, "g", ".ticks").style("font", "15px " + font).attr("transform", "translate(0," + 18 + ")").selectAll("text").data(tickData).enter().append("text").attr("x", function (d) {
      return xScale(d.pos);
    }).attr("text-anchor", "middle").text(function (d) {
      return d.label;
    });

    // write the label
    (0, _selectAppend2.default)(slider, "text", ".label").attr("y", -14).attr("font-family", font).text(label);

    // apply styles to everything.
    (0, _styles.roundEndsStyle)(track);
    (0, _styles.trackStyle)(track);
    (0, _styles.handleStyle)(handle);
    (0, _styles.roundEndsStyle)(trackOverlay);
    (0, _styles.trackOverlayStyle)(trackOverlay);
    (0, _styles.roundEndsStyle)(trackInset);
    (0, _styles.trackInsetStyle)(trackInset);

    // setup callbacks
    // Calculates the beta function between alpha and beta
    /**
    * @return {object} - A slider
    */
    function dragBehavior() {
      var scaledPos = getValue(d3.event.x);
      // by inverting and reverting the position we assert bounds on the slider.
      handle.attr("cx", xScale(scaledPos));
      onDrag ? onDrag(scaledPos) : null;
    }

    // Calculates the beta function between alpha and beta
    /**
    * @return {object} - A slider
    */
    function finishBehavior() {
      var dragPos = getValue(d3.event.x);
      var finalPos = intClamp ? Math.round(dragPos) : dragPos;
      var closestTickColor = (0, _findClosestTickColor2.default)(tickData, finalPos);
      handle.transition(trans).attr("cx", xScale(finalPos)).attr("fill", closestTickColor);
      onDone(finalPos);
    }
  } // end drawSlider()

  // Getter and setters for changing settings.

  drawSlider.range = function (range) {
    if (!arguments.length) return sliderRange;
    sliderRange = range;
    return drawSlider;
  };

  drawSlider.width = function (width) {
    if (!arguments.length) return sliderWidth;
    sliderWidth = width;
    return drawSlider;
  };

  drawSlider.onDone = function (doneFunc) {
    if (!arguments.length) return onDone;
    onDone = doneFunc;
    return drawSlider;
  };

  drawSlider.onDrag = function (dragFunc) {
    if (!arguments.length) return onDrag;
    onDrag = dragFunc;
    return drawSlider;
  };

  drawSlider.label = function (newLabel) {
    if (!arguments.length) return label;
    label = newLabel;
    return drawSlider;
  };

  drawSlider.startPos = function (newStartPos) {
    if (!arguments.length) return startPos;
    startPos = newStartPos;
    return drawSlider;
  };

  drawSlider.loc = function (loc) {
    if (!arguments.length) return [xPos, yPos];

    var _loc = _slicedToArray(loc, 2);

    xPos = _loc[0];
    yPos = _loc[1];

    return drawSlider;
  };

  drawSlider.clamp = function (decision) {
    if (!arguments.length) return intClamp;
    intClamp = decision;
    return drawSlider;
  };

  drawSlider.vertical = function (decision) {
    if (!arguments.length) return vertical;
    vertical = decision;
    return drawSlider;
  };

  drawSlider.numTicks = function (num) {
    if (!arguments.length) return numTicks;
    numTicks = num;
    return drawSlider;
  };

  drawSlider.customTicks = function (tickLabels) {
    if (!arguments.length) return customTicks;
    customTicks = tickLabels;
    return drawSlider;
  };

  drawSlider.handleColor = function (color) {
    if (!arguments.length) return handleColor;
    handleColor = color;
    return drawSlider;
  };

  drawSlider.font = function (newFont) {
    if (!arguments.length) return font;
    font = newFont;
    return drawSlider;
  };

  drawSlider.animation = function (speed) {
    transitionSpeed = speed ? speed : 0; // allow the user to have passed something like 'false' to this.
    if (!arguments.length) return transitionSpeed;
    return drawSlider;
  };

  return drawSlider;
}

module.exports = slid3r;

/***/ }),
/* 1 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});

// tries to do a selection, if it's empty, deals with it by appending desired.

exports.default = function (parent, type, identifier) {
  var attemptedSelection = parent.select("" + type + identifier);
  var emptySelection = attemptedSelection.empty();
  var identifierType = identifier.charAt(0) == "." ? "class" : "id";
  return emptySelection ? parent.append(type).attr(identifierType, identifier) : attemptedSelection;
};

/***/ }),
/* 2 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});

exports.default = function (tickData, finalPos) {
  var closestTick = tickData.reduce(function (closest, current, i) {
    var distanceFromTick = Math.abs(finalPos - current.pos);
    return distanceFromTick < closest.distance || closest.distance === null ? { distance: distanceFromTick, index: i } : closest;
  }, { distance: null, index: -1 });

  return tickData[closestTick.index].color;
};

/***/ }),
/* 3 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});
// styles for the d3 slider

// styles
var roundEndsStyle = exports.roundEndsStyle = function roundEndsStyle(selection) {
  return selection.style('stroke-linecap', 'round');
};

var trackStyle = exports.trackStyle = function trackStyle(selection) {
  return selection.style('stroke', '#000').style('stroke-opacity', '0.3').style('strokeWidth', '10px');
};

var trackInsetStyle = exports.trackInsetStyle = function trackInsetStyle(selection) {
  return selection.style('stroke', '#ddd').style('stroke-width', 8);
};

var trackOverlayStyle = exports.trackOverlayStyle = function trackOverlayStyle(selection) {
  return selection.style('pointer-events', 'stroke').style('stroke-width', 50).style('stroke', 'transparent').style('cursor', 'crosshair');
};

var handleStyle = exports.handleStyle = function handleStyle(selection) {
  return selection
  // .style('fill', '#fff')
  .style('stroke', '#000').style('stroke-opacity', 0.5).style('strokeWidth', '1.25px');
};

/***/ }),
/* 4 */
/***/ (function(module, exports) {

module.exports = __WEBPACK_EXTERNAL_MODULE_4__;

/***/ })
/******/ ]);
});