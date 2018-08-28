import React, { PropTypes } from 'react';
import ReactDom from 'react-dom';
import { connect } from 'react-redux'

class NetStatistics extends React.Component {

  constructor(props) {
    super(props);
    this.createNodeValuesChart = this.createNodeValuesChart.bind(this);
    // this.state = DEFAULT_STATE;
  }

  componentDidMount(){
  };

  onClearButtonClick(){
  };

  createNodeValuesChart() {
    const netStatistics = this.props.netStatistics;
    console.log("nodeValues: ", netStatistics)
    return(
      netStatistics.map( (val, i) => {
        return <div key={i}> { val }</div>
      })
    )
  }

  render() {
    const name = this.props.name;
    const nodeValues = this.props.nodeValues;
    const prediction = this.props.prediction;

    return (
      <div className="net-statistics">
        <div>
          name: { name }
        </div>
        <div>
          { this.createNodeValuesChart() }
        </div>
        <div>
          prediction: { prediction }
        </div>
      </div>
    );
  }
};

const mapStateToProps = (state) => {
  return {
    netStatistics: state.netStatistics
  }
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
