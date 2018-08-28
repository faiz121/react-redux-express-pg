import React, { PropTypes } from 'react';
import ReactDom from 'react-dom';
import { connect } from 'react-redux'

class NetStatistics extends React.Component {

  constructor(props) {
    super(props);
    // this.state = DEFAULT_STATE;
  }

  componentDidMount(){
  };

  onClearButtonClick(){
  };

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

const mapStateToProps = (state) => {
  return { }
};

NetStatistics.defaultProps = { };

const mapDispatchToProps = (dispatch) => {
  return {
    // dispatchAddTodo(todo) {
    //   dispatch(postTodoToDB(todo))
    // }
  }
};

export default connect(mapStateToProps, mapDispatchToProps)(NetStatistics);
