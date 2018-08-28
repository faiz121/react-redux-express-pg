import React, { PropTypes } from 'react';
import ReactDom from 'react-dom';
import { connect } from 'react-redux'

class OneHotResultChart extends React.Component {

  constructor(props) {
    super(props);
    console.log("props: ", props)
    this.createOneHotResultChartChart = this.createOneHotResultChartChart.bind(this);
  }

  componentDidMount(){
  };

  onClearButtonClick(){
  };

  createOneHotResultChartChart() {
    const netStatistics = this.props.netStatistics;
    console.log("nodeValues: ", netStatistics)
    return(
      netStatistics.map( (netStat, i) => {
        return <div key={i}> { val }</div>
      })
    )
  }

  render() {
    return (
      <div className="net-statistics">
          {
            this.props.oneHotResult.map( (ohr, i) => {
              return <div key={"ohr" + i}> {ohr} </div>
            })
          }
      </div>
    );
  }
};

const mapStateToProps = (state) => {
  return {
    netStatistics: state.netStatistics
  }
};

OneHotResultChart.defaultProps = { };

const mapDispatchToProps = (dispatch) => {
  return {
    // dispatchAddTodo(todo) {
    //   dispatch(postTodoToDB(todo))
    // }
  }
};

export default connect(mapStateToProps, mapDispatchToProps)(OneHotResultChart);
