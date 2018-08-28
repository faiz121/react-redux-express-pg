import React, { PropTypes } from 'react';
import ReactDom from 'react-dom';
import { connect } from 'react-redux'

// const DEFAULT_STATE = {
//   'name': '',
//   'nodeValues': [],
//   'prediction': 0
// }

class NetStatistics extends React.Component {

  constructor(props) {
    super(props);
    // this.state = DEFAULT_STATE;
  }

  componentDidMount(){
  }

  onClearButtonClick(){
    var ctx = this.state.canvas.getContext('2d');
    ctx.clearRect(0, 0, this.state.canvas.width, this.state.canvas.height);
  }

  render() {
    const name = this.props.name;
    const nodeValues = this.props.nodeValues;
    const prediction = this.props.prediction;

    return (
      <div className="net-statistics">
        { name }
        { nodeValues }
        { prediction }
      </div>
    );
  }
};

// NetStatistics.propTypes = {
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

NetStatistics.defaultProps = {
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

export default connect(mapStateToProps, mapDispatchToProps)(NetStatistics);
