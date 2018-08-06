import React, { PropTypes } from 'react';
import ReactDom from 'react-dom';
import { connect } from 'react-redux'
import { SliderPicker } from 'react-color';
import axios from 'axios'

class DrawCanvas extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      canvas: null,
      context: null,
      drawing: false,
      lastX: 0,
      lastY: 0,
      history: [],
      brushColor: '#0000ff'
    };
    // this.handleMouseDown = this.handleMouseDown.bind(this)
    // this.handleMouseUp = this.handleMouseUp.bind(this)
    // this.onClick = this.onClick.bind(this)
    // this.renderGraphic = this.renderGraphic.bind(this)
  }

  handleChangeComplete(color){
    this.setState({
      brushColor: color.hex,
      clear: false
    });
  }

  componentDidMount(){
    // var canvas = ReactDom.findDOMNode(this);
    var canvas = this.refs.canvas;

    canvas.style.width = '112px';
    canvas.style.height = '112px';
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    var ctx = canvas.getContext('2d');

    this.setState({
      canvas: canvas,
      context: ctx
    });
  }

  _onDrawingEvent(data){

    var w = this.state.context.canvas.width
    var h = this.state.context.canvas.height
    this.draw(data.lX*w, data.lY*h, data.cX*w, data.cY*h, data.brushColor);
  }

  _onCatchUpEvent(history){
    var w = this.state.context.canvas.width
    var h = this.state.context.canvas.height
    for(var i = 0; i < history.length; i++){
      var currentLine = history[i];
      this.draw(currentLine.lX*w, currentLine.lY*h, currentLine.cX*w, currentLine.cY*h, currentLine.brushColor);
    }
  }

  throttle(callback, delay) {
    var previousCall = new Date().getTime();
    return function() {
      var time = new Date().getTime();

      if ((time - previousCall) >= delay) {
        previousCall = time;
        callback.apply(null, arguments);
      }
    };
  }

  componentWillReceiveProps(nextProps) {
    if(nextProps.clear){
      this.resetCanvas();
    }
  }
  handleOnMouseDown(e){
    var rect = this.state.canvas.getBoundingClientRect();
    this.state.context.beginPath();

    this.setState({
      lastX: e.clientX - rect.left,
      lastY: e.clientY - rect.top
    });

    this.setState({
      drawing: true
    });

    console.log("on mouse down event")


  }
  handleOnMouseMove(e){

    if(this.state.drawing){
      var rect = this.state.canvas.getBoundingClientRect();
      var lastX = this.state.lastX;
      var lastY = this.state.lastY;
      var currentX;
      var currentY;
      if(this.isMobile()){
        currentX =  e.targetTouches[0].pageX - rect.left;
        currentY = e.targetTouches[0].pageY - rect.top;
      }
      else{
        currentX = e.clientX - rect.left;
        currentY = e.clientY - rect.top;
      }


      this.draw(lastX, lastY, currentX, currentY, this.state.brushColor);
      this.setState({
        lastX: currentX,
        lastY: currentY
      });
    }
  }
  handleonMouseUp(){
    this.setState({
      drawing: false
    });
  }
  draw(lX, lY, cX, cY, brushColor){

    this.state.context.strokeStyle = brushColor;
    this.state.context.lineWidth = this.props.lineWidth;
    this.state.context.beginPath();
    this.state.context.moveTo(lX,lY);
    this.state.context.lineTo(cX,cY);
    this.state.context.stroke();

    var w = this.state.context.canvas.width
    var h = this.state.context.canvas.height

    this.state.history.push({
      lX: lX/w,
      lY: lY/h,
      cX: cX/w,
      cY: cY/h,
      brushColor: brushColor
    })

  }
  resetCanvas(){
    var width = this.state.context.canvas.width;
    var height = this.state.context.canvas.height;
    this.state.context.clearRect(0, 0, width, height);
  }
  getDefaultStyle(){
    return {
      backgroundColor: '#FFFFFF',
      cursor: 'pointer'
    };
  }
  canvasStyle(){
    var defaults =  this.getDefaultStyle();
    var custom = this.props.canvasStyle;
    return Object.assign({}, defaults, custom);
  }
  isMobile(){
    if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
      return true;
    }
    return false;
  }
  onButtonClick(){
    var ctx = this.state.canvas.getContext('2d');
    var dataUrl = this.state.canvas.toDataURL("image/png")

    console.log("Data url: ", dataUrl);

    axios.get(`http://localhost:4002/process_image?dataUrl=${encodeURIComponent(dataUrl)}`)
      .then(function (response) {
        console.log("data back from backend: ", response);
      })
      .catch(function (error) {
        console.log(error);
      });
  }

  render() {
    return (
      <div>
        <h2>Drawing Canvas </h2>
        <SliderPicker
          color={ this.state.brushColor }
          onChangeComplete={ this.handleChangeComplete.bind(this) }
        />
        <canvas id="drawCanvas" ref='canvas' style={{ borderStyle: 'solid', borderWidth: '5px', backgroundColor: '#FFFFFF', cursor: 'pointer'}}
          onMouseDown  =  { this.handleOnMouseDown.bind(this) }
          onTouchStart =  { this.handleOnMouseDown.bind(this) }
          onMouseMove  =  { this.handleOnMouseMove.bind(this) }
          onTouchMove  =  { this.handleOnMouseMove.bind(this) }
          onMouseUp    =  { this.handleonMouseUp.bind(this) }
          onTouchEnd   =  { this.handleonMouseUp.bind(this) }
        >
        </canvas>
        <button type="button" onClick = { this.onButtonClick.bind(this) } >
          Click Me!
        </button>
      </div>
    );
  }
};

// DrawCanvas.propTypes = {
//   lineWidth: PropTypes.number,
//   canvasStyle: PropTypes.shape({
//     backgroundColor: PropTypes.string,
//     cursor: PropTypes.string
//   }),
//   clear: PropTypes.bool
// }

const mapStateToProps = (state) => {
  return {
    image: state.image
  }
};

DrawCanvas.defaultProps = {
  lineWidth: 4,
  canvasStyle: {
    backgroundColor: '#FFFFFF',
    cursor: 'pointer'
  },
  clear: false
}

const mapDispatchToProps = (dispatch) => {
  // console.log(`setSearchTerm ${setSearchTerm}`);
  return {
    // dispatchSetSearchTerm (searchTerm) {
    //   dispatch(setSearchTerm(searchTerm)) // dispatch({ type: 'SET_SEARCH_TERM', searchTerm = searchTerm })
    // },
    // dispatchAddTodo(todo) {
    //   dispatch(postTodoToDB(todo))
    // }
  }
};

export default connect(mapStateToProps, mapDispatchToProps)(DrawCanvas);